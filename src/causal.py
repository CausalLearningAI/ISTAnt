def compute_ead(Y, T, verbose=True):
    E_Y_t0 = Y[T==0].mean()
    E_Y_t1 = Y[T==1].mean()
    E_Y_t2 = Y[T==2].mean()
    EAD_B = E_Y_t1 - E_Y_t0
    if verbose: 
        print(f"EAD_B: {EAD_B:.3f} (+{(EAD_B)/E_Y_t0*100:.0f}%)")
    EAD_inf = E_Y_t2 - E_Y_t0
    if verbose: 
        print(f"EAD_inf: {EAD_inf:.3f} (+{(EAD_inf)/E_Y_t0*100:.0f}%)")
    return EAD_B, EAD_inf