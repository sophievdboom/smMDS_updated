import traceback

print("Testing diffusion_device import...")

try:
    import diffusion_device
    print("OK: imported diffusion_device")
except Exception:
    print("FAILED: import diffusion_device")
    traceback.print_exc()
    raise

print("Testing key submodules...")

modules_to_test = [
    "diffusion_device.process_data",
    "diffusion_device.keys",
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"OK: imported {module_name}")
    except Exception:
        print(f"FAILED: import {module_name}")
        traceback.print_exc()

print("Done.")