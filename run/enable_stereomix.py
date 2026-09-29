"""Enable the Windows 'Stereo Mix' capture endpoint via COM (WASAPI).

Uses the MMDeviceEnumerator COM interface to find the 'Stereo Mix' capture
device and call SetState(eEnabled). This requires no admin for the user's
own audio session on most setups (may prompt UAC).

Run:  python run/enable_stereomix.py
"""
import ctypes
import ctypes.wintypes as wt
import sys

# GUID helper (ctypes doesn't have GUID on all builds)
class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wt.DWORD),
        ("Data2", ct_word := ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    ]

def parse_guid(s: str) -> GUID:
    s = s.strip("{}")
    parts = s.split("-")
    d1 = int(parts[0], 16)
    d2 = int(parts[1], 16)
    d3 = int(parts[2], 16)
    d4_bytes = bytes.fromhex(parts[3] + parts[4])
    d4 = (ctypes.c_uint8 * 8)(*d4_bytes)
    return GUID(d1, d2, d3, d4)

CLSID_MMDeviceEnumerator = parse_guid("BCDE0395-E52F-467C-8E3D-C46788A9F57E")
IID_IMMDeviceEnumerator = parse_guid("A9566492-95D1-4925-A1B2-97B84F71FB29")
IID_IMMDevice = parse_guid("D6663630-3410-413F-917B-35C61089C47B")

EVT_RENDER = 0
EVT_CAPTURE = 1
eEnabled = 1
eDisabled = 2
eUnplugged = 4

# CoCreateInstance(rclsid, pUnkOuter, dwClsContext, riid, ppv)
# All COM pointers are c_void_p; GUIDs are passed as addresses (byref)
CoCreateInstance = ctypes.windll.ole32.CoCreateInstance
CoCreateInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
CoCreateInstance.restype = ctypes.c_long

def hr_ok(hr, what):
    if hr != 0:
        print(f"  {what}: HRESULT 0x{hr & 0xffffffff:08x}")
        return False
    return True

def main():
    clsid_ptr = ctypes.byref(CLSID_MMDeviceEnumerator)
    iid_ptr = ctypes.byref(IID_IMMDeviceEnumerator)
    obj = ctypes.c_void_p()
    # CoCreateInstance(rclsid, pUnkOuter, dwClsContext, riid, ppv)
    # Pass GUIDs as pointers to their addresses
    hr = CoCreateInstance(clsid_ptr, None, 1, iid_ptr, ctypes.byref(obj))
    if not hr_ok(hr, "CoCreateInstance"):
        return 1
    # vtable: IUnknown(3) + GetDefaultAudioEndpoint(4) + GetDevice(5) + EnumerateAudioEndpoints(6)
    vt = ctypes.cast(ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p)).contents,
                     ctypes.POINTER(ctypes.c_void_p)).contents
    EnumerateAudioEndpoints = vt[6]
    EnumerateAudioEndpoints.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                        ctypes.POINTER(ctypes.c_void_p)]
    EnumerateAudioEndpoints.restype = ctypes.HRESULT
    coll = ctypes.c_void_p()
    hr = EnumerateAudioEndpoints(None, EVT_CAPTURE, ctypes.byref(coll))
    if not hr_ok(hr, "EnumerateAudioEndpoints"):
        return 1
    # IEnumUnknown vtable: IUnknown(3) + Next(4) + Skip(5) + Reset(6) + Clone(7)
    cvt = ctypes.cast(ctypes.cast(coll, ctypes.POINTER(ctypes.c_void_p)).contents,
                      ctypes.POINTER(ctypes.c_void_p)).contents
    Next = cvt[4]
    Next.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                     ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    Next.restype = ctypes.HRESULT
    print("Capture endpoints:")
    found_stereo = False
    n = 0
    while n < 40:
        pdev = ctypes.c_void_p()
        got = ctypes.c_void_p()
        hr = Next(None, 1, ctypes.byref(pdev), ctypes.byref(got))
        if hr != 0 or not pdev.value:
            break
        # IMMDevice vtable: IUnknown(3) + Activate(4) + GetId(5) + GetState(6) + ... + SetState(10)
        dvt = ctypes.cast(ctypes.cast(pdev, ctypes.POINTER(ctypes.c_void_p)).contents,
                          ctypes.POINTER(ctypes.c_void_p)).contents
        GetId = dvt[5]
        GetId.argtypes = [ctypes.c_void_p, ctypes.POINTER(wt.LPWSTR)]
        GetId.restype = ctypes.HRESULT
        GetState = dvt[6]
        GetState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        GetState.restype = ctypes.HRESULT
        SetState = dvt[10]
        SetState.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        SetState.restype = ctypes.HRESULT
        idbuf = wt.LPWSTR()
        GetId(None, ctypes.byref(idbuf))
        name = ctypes.cast(idbuf, ctypes.c_wchar_p).value if idbuf else "?"
        state = ctypes.c_uint32()
        GetState(None, ctypes.byref(state))
        enabled = (state.value & eEnabled) != 0
        marker = ""
        if "Стерео микшер" in name or "stereo mix" in name.lower():
            marker = "  <-- STEREO MIX"
            found_stereo = True
            if not enabled:
                print(f"  [{n}] state={state.value} enabled={enabled} {name} {marker}")
                print("  -> attempting SetState(eEnabled)...")
                hr = SetState(None, eEnabled)
                if hr_ok(hr, "SetState"):
                    # re-read state
                    GetState(None, ctypes.byref(state))
                    print(f"  -> new state={state.value} (1=enabled)")
                else:
                    print("  -> SetState failed (may need admin / UAC)")
            else:
                print(f"  [{n}] state={state.value} enabled={enabled} {name} {marker} (already enabled)")
        else:
            print(f"  [{n}] state={state.value} enabled={enabled} {name}")
        n += 1
    print(f"\nStereo Mix found: {found_stereo}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
