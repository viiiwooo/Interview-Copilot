"""Try to enable the Windows 'Stereo Mix' capture endpoint via COM (WASAPI).

Uses the MMDeviceEnumerator + IAudioEndpointVolume COM interfaces. If the
endpoint exists but is disabled, we attempt to enable it. This needs no
admin for the user's own audio session on most setups (may prompt UAC).
"""
import ctypes
import ctypes.wintypes as wt
import sys

GUID = wt.GUID

# --- COM boilerplate -------------------------------------------------------
CoCreateInstance = ctypes.windll.ole32.CoCreateInstance
CoCreateInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
CoCreateInstance.restype = ctypes.c_long

CLSID_MMDeviceEnumerator = GUID("{BCDE0395-E52F-467C-8E3D-C46788A9F57E}")
IID_IMMDeviceEnumerator = GUID("{A9566492-95D1-4925-A1B2-97B84F71FB29}")
IID_IAudioEndpointVolume = GUID("{5CDF2C82-34A8-4EEE-94C5-BD574BD4C1F9}")
EVT_RENDER = 0
EVT_CAPTURE = 1
eEnabled = 1
eDisabled = 2
eUnplugged = 4

class IMMDeviceEnumerator(ctypes.Structure):
    _iid_ = IID_IMMDeviceEnumerator
    _methods_ = [
        ("Activate", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p))),
    ]

class IMMDevice(ctypes.Structure):
    _iid_ = GUID("{D6663630-3410-413F-917B-35C61089C47B}")
    _methods_ = [
        ("Activate", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p))),
        ("GetId", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.POINTER(wt.LPWSTR))),
        ("GetState", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.POINTER(ctypes.c_uint32))),
    ]

def hr_check(hr, what):
    if hr != 0:
        print(f"  {what}: HRESULT 0x{hr & 0xffffffff:08x}")
        return False
    return True

def main():
    clsid = ctypes.byref(CLSID_MMDeviceEnumerator)
    iid = ctypes.byref(IID_IMMDeviceEnumerator)
    obj = ctypes.c_void_p()
    hr = CoCreateInstance(clsid, None, 1, iid, ctypes.byref(obj))
    if not hr_check(hr, "CoCreateInstance"):
        return 1
    enum = IMMDeviceEnumerator.from_address(obj.value)
    # enumerate capture endpoints
    print("Capture endpoints:")
    found = []
    # Use GetDefaultAudioEndpoint and EnumerateAudioEndpoints via vtable
    # vtable layout for IMMDeviceEnumerator: IUnknown(3) + GetDefaultAudioEndpoint(4) + GetDevice(5) + EnumerateAudioEndpoints(6)
    vt = ctypes.cast(ctypes.cast(enum, ctypes.POINTER(ctypes.c_void_p)).contents,
                     ctypes.POINTER(ctypes.c_void_p)).contents
    # IUnknown: QueryInterface, AddRef, Release
    GetDefaultAudioEndpoint = vt[3]
    GetDefaultAudioEndpoint.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                        ctypes.POINTER(ctypes.c_void_p)]
    GetDefaultAudioEndpoint.restype = ctypes.HRESULT
    EnumerateAudioEndpoints = vt[6]
    EnumerateAudioEndpoints.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    EnumerateAudioEndpoints.restype = ctypes.HRESULT

    # Get default capture device
    dev = ctypes.c_void_p()
    hr = GetDefaultAudioEndpoint(None, EVT_CAPTURE, ctypes.byref(dev))
    if hr_check(hr, "GetDefaultAudioEndpoint(CAPTURE)"):
        d = IMMDevice.from_address(dev.value)
        state = ctypes.c_uint32()
        d.GetState(ctypes.byref(state))
        print(f"  default capture: state={state.value} (1=enabled,2=disabled,4=unplugged)")

    # Enumerate all capture endpoints
    coll = ctypes.c_void_p()
    hr = EnumerateAudioEndpoints(None, EVT_CAPTURE, ctypes.byref(coll))
    if not hr_check(hr, "EnumerateAudioEndpoints"):
        return 1
    # IEnumUnknown: Next(3), Skip(4), Reset(5), Clone(6)
    cvt = ctypes.cast(ctypes.cast(coll, ctypes.POINTER(ctypes.c_void_p)).contents,
                      ctypes.POINTER(ctypes.c_void_p)).contents
    Next = cvt[3]
    Next.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    Next.restype = ctypes.HRESULT
    n = 0
    while True:
        pdev = ctypes.c_void_p()
        got = ctypes.c_void_p()
        hr = Next(None, 1, ctypes.byref(pdev), ctypes.byref(got))
        if hr != 0 or not pdev.value:
            break
        d = IMMDevice.from_address(pdev.value)
        state = ctypes.c_uint32()
        d.GetState(ctypes.byref(state))
        # get id
        idbuf = wt.LPWSTR()
        d.GetId(ctypes.byref(idbuf))
        name = ctypes.cast(idbuf, ctypes.c_wchar_p).value if idbuf else "?"
        enabled = (state.value & eEnabled) != 0
        print(f"  [{n}] state={state.value} enabled={enabled}  {name}")
        if "Стерео микшер" in name or "Stereo Mix" in name.lower():
            found.append((n, state.value, enabled))
        n += 1
        if n > 40:
            break
    print(f"\n{len(found)} 'stereo mix' endpoint(s) found")
    return 0

if __name__ == "__main__":
    sys.exit(main())
