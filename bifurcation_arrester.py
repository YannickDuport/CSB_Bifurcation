import numpy as np
import PyDSTool as dst
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp

save = True     # set to True if you want to save the bifurcation diagram
matplotlib.rc('font', **{'size': 13})

def main():
    # define ode system
    def arrester(t, y, p):

        p53_arrester, Rb_tot, E2F1_tot, q0_p21, q1_p21, q2, s5, t5, s9, s10, p9, p10, d12, b4, b5, u5, u6, g5, g19, g20, h, M2, M3 = p
        p21_mRNA, p21, Rb_0, Rb_0_E2F1_cx, CyclinE, CyclinE_p21_cx = y

        dp21_mRNAdt = s5 * ((q0_p21 + q1_p21 * np.power(p53_arrester, h)) / (q2 + q0_p21 + q1_p21 * np.power(p53_arrester, h)))- \
                      g5 * p21_mRNA
        dp21dt = t5 * p21_mRNA + u6 * CyclinE_p21_cx - b5 * CyclinE * p21 - g19 * p21
        dRb_0dt = (d12 * (Rb_tot - (Rb_0 + Rb_0_E2F1_cx)) / (M2 + (Rb_tot - (Rb_0 + Rb_0_E2F1_cx)))) - \
                  b4 * Rb_0 * (E2F1_tot - Rb_0_E2F1_cx) - p9 * CyclinE * Rb_0 + u5 * Rb_0_E2F1_cx
        dRb_0_E2F1_cxdt = b4 * Rb_0 * (E2F1_tot - Rb_0_E2F1_cx) - u5 * Rb_0_E2F1_cx - p10 * CyclinE * Rb_0_E2F1_cx
        dCyclinEdt = s10 + s9 * np.power(E2F1_tot - Rb_0_E2F1_cx, 2) / (np.power(M3, 2) + np.power(E2F1_tot-Rb_0_E2F1_cx, 2)) -\
                     b5 * CyclinE * p21 + u6 * CyclinE_p21_cx - g20 * CyclinE
        dCyclinE_p21_cxdt = b5 * CyclinE * p21 - u6 * CyclinE_p21_cx - g20 * CyclinE_p21_cx

        return np.array([dp21_mRNAdt, dp21dt, dRb_0dt, dRb_0_E2F1_cxdt, dCyclinEdt, dCyclinE_p21_cxdt])

    # set parameter values
    p53_arrester = 10000
    Rb_tot = 3e5
    E2F1_tot = 2e5
    q0_p21 = 1e-5
    q1_p21 = 3e-13
    q2 = 3e-3
    s5 = 0.1
    t5 = 0.1
    s9 = 30
    s10 = 3
    p9 = 3e-6
    p10 = p9
    d12 = 1e4
    b4 = 1e-5
    b5 = 1e-5
    u5 = 1e-4
    u6 = 1e-4
    g5 = 3e-4
    g19 = 3e-4
    g20 = 1e-4
    h = 2
    M2 = 1e5
    M3 = 2e5
    params = [p53_arrester, Rb_tot, E2F1_tot, q0_p21, q1_p21,
              q2, s5, t5, s9, s10, p9, p10, d12, b4, b5,
              u5, u6, g5, g19, g20, h, M2, M3]

    # set initial values
    p21_mRNA = 0
    p21 = 0
    Rb_0 = 0
    Rb_0_E2F1_cx = 0
    CyclinE = 0
    CyclinE_p21_cx = 0
    x0 = np.array([p21_mRNA, p21, Rb_0, Rb_0_E2F1_cx, CyclinE, CyclinE_p21_cx])


    ### FIND INITIAL STEADY STATE ###
    print("Determine initial steady state(s)...")
    # set simulation time (end point should be large, so the system is in steady state)
    eq_step = 1e4
    eq_duration = 1e7
    t_range = (0, eq_duration)

    # solve initial value problem (simulate system)
    method = "LSODA"
    eq = solve_ivp(arrester, t_range, x0, args=(params,),
                   method=method, t_eval=[eq_duration]).y


    ### CREATE BIFURCATION DIAGRAM ###

    ### 1) Set up model for PyDSTools
    # Map parameter values to names
    p53_arrester = dst.Par(10000, 'p53_arrester')
    Rb_tot = dst.Par(3e5, 'Rb_tot')
    F1_tot = dst.Par(2e5, 'F1_tot')
    q0_p21 = dst.Par(1e-5, 'q0_p21')
    q1_p21 = dst.Par(3e-13, 'q1_p21')
    q2 = dst.Par(3e-3, 'q1')
    s5 = dst.Par(0.1, 's5')
    t5 = dst.Par(0.1, 't5')
    s9 = dst.Par(30, 's9')
    s10 = dst.Par(3, 's10')
    p9 = dst.Par(3e-6, 'p9')
    p10 = dst.Par(3e-6, 'p10')
    d12 = dst.Par(1e4, 'd12')
    b4 = dst.Par(1e-5, 'b4')
    b5 = dst.Par(1e-5, 'b5')
    u5 = dst.Par(1e-4, 'u5')
    u6 = dst.Par(1e-4, 'u6')
    g5 = dst.Par(3e-4, 'g5')
    g19 = dst.Par(3e-4, 'g19')
    g20 = dst.Par(1e-4, 'g20')
    h = dst.Par(2, 'h')
    M2 = dst.Par(1e5, 'M2')
    M3 = dst.Par(2e5, 'M3')

    # define variables
    p21_mRNA = dst.Var('p21_mRNA')
    p21 = dst.Var('p21')
    Rb_0 = dst.Var('Rb_0')
    Rb_0_E2F1_cx = dst.Var('Rb_0_E2F1_cx')
    CyclinE = dst.Var('CyclinE')
    CyclinE_p21_cx = dst.Var('CyclinE_p21_cx')

    # define reactions
    p21_mRNA_rhs = s5 * ((q0_p21 + q1_p21 * p53_arrester**h) / (q2 + q0_p21 + q1_p21 * p53_arrester**h)) - g5 * p21_mRNA
    p21_rhs = t5 * p21_mRNA + u6 * CyclinE_p21_cx - b5 * CyclinE * p21 - g19 * p21
    Rb_0_rhs = (d12 * (Rb_tot - (Rb_0 + Rb_0_E2F1_cx)) / (M2 + (Rb_tot - (Rb_0 + Rb_0_E2F1_cx)))) - b4 * Rb_0 * (F1_tot - Rb_0_E2F1_cx) - p9 * CyclinE * Rb_0 + u5 * Rb_0_E2F1_cx
    Rb_0_E2F1_cx_rhs = b4 * Rb_0 * (F1_tot - Rb_0_E2F1_cx) - u5 * Rb_0_E2F1_cx - p10 * CyclinE * Rb_0_E2F1_cx
    CyclinE_rhs = s10 + s9 * (F1_tot - Rb_0_E2F1_cx)**2 / ((M3**2) + (F1_tot - Rb_0_E2F1_cx)**2) - b5 * CyclinE * p21 + u6 * CyclinE_p21_cx - g20 * CyclinE
    CyclinE_p21_cx_rhs = b5 * CyclinE * p21 - u6 * CyclinE_p21_cx - g20 * CyclinE_p21_cx

    # Create model
    DSargs = dst.args(name='arrester')                  # model name
    DSargs.pars = [                                     # model parameters
        p53_arrester, Rb_tot, F1_tot, q0_p21, q1_p21,
        q2, s5, t5, s9, s10, p9, p10, d12, b4, b5,
        u5, u6, g5, g19, g20, h, M2, M3
    ]
    DSargs.varspecs = dst.args(                         # ODEs
        p21_mRNA=p21_mRNA_rhs,
        p21=p21_rhs,
        Rb_0=Rb_0_rhs,
        Rb_0_E2F1_cx=Rb_0_E2F1_cx_rhs,
        CyclinE=CyclinE_rhs,
        CyclinE_p21_cx=CyclinE_p21_cx_rhs
    )
    DSargs.ics = dst.args(                              # initial values
        p21_mRNA=eq[0],
        p21=eq[1],
        Rb_0=eq[2],
        Rb_0_E2F1_cx=eq[3],
        CyclinE=eq[4],
        CyclinE_p21_cx=eq[5]
    )

    # Create ODE System for PyDSTools
    ode = dst.Generator.Vode_ODEsystem(DSargs)


    ### 2) Numerical continuation
    # create empty figure
    fig = plt.figure(figsize=(7, 6))

    # define problem
    PC = dst.ContClass(ode)
    PCargs = dst.args(name='EQ1', type='EP-C')          # name and type (equilibrium point curve)
    PCargs.freepars = ['p53_arrester']                  # set free parameter
    PCargs.MaxNumPoints = 1000
    PCargs.MaxStepSize = 500
    PCargs.MinStepSize = 1
    PCargs.StepSize = 25
    PCargs.LocBifPoints = 'LP'                          # detect limit points (saddle-node)
    PCargs.SaveEigen = True                             # compute and show stability

    # compute a new equilibrium point curve
    print("Numerical Continuation...")
    PC.newCurve(PCargs)
    PC['EQ1'].backward()

    # draw bifurcation diagram (p53_arrester vs. CyclinE)
    PC.display(['p53_arrester', 'CyclinE'], stability=True, figure=fig)

    # edit figure
    plt.xlabel(r'$\mathregular{p53_{ARRESTER}}$', fontsize=13)
    plt.tick_params(labelsize=10)
    plt.title("")
    fig.tight_layout()
    plt.show()
    if save:
        fig.savefig('bifurcation_arrest.png', dpi=300)


if __name__ == "__main__":
    main()
