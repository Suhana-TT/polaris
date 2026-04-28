import ttsim.front.ttnn as ttnn

print("Has slice:", hasattr(ttnn, 'slice'))

# List all slice-related functions
slice_ops = [op for op in dir(ttnn) if 'slice' in op.lower()]
print("Slice operations:", slice_ops)

# If slice exists, check its signature
if hasattr(ttnn, 'slice'):
    print("\nslice exists! Checking signature...")
    import inspect
    try:
        print(inspect.signature(ttnn.slice))
    except:
        pass
    help(ttnn.slice)