import numpy as np
from scipy.integrate import solve_ivp
import PyDSTool as dst
import matplotlib
import matplotlib.pyplot as plt


save = True     # set to True if you want to save the bifurcation diagram
matplotlib.rc('font', **{'size': 13})

def main():
    # define ode system
    def apoptosis(t, y, p):

        p53_killer, AKT_p, Bad_tot, BclXL_tot, Fourteen33_tot, a1, a2, q0_bax, q1_bax, q2, s4, t4, s7, p7, d9, b1, b2, b3, u1, u2, u3, g4, g9, g16, g17, g18, h = p
        Bax_mRNA, Bax, BclXL, Bax_BclXL_cx, Bad_0, Bad_p, proCasp, Casp = y

        dBax_mRNAdt = s4 * ((q0_bax + q1_bax * p53_killer**h) / (q2 + q0_bax + q1_bax * p53_killer**h)) - g4 * Bax_mRNA
        dBaxdt = t4 * Bax_mRNA + u1 * Bax_BclXL_cx - b1 * BclXL * Bax - g9 * Bax
        dBclXLdt = u2 * (BclXL_tot - (BclXL + Bax_BclXL_cx)) + u1 * Bax_BclXL_cx + g16 * Bax_BclXL_cx + \
                   p7 * AKT_p * (BclXL_tot - (BclXL + Bax_BclXL_cx)) - b2 * BclXL * Bad_0 - b1 * BclXL * Bax
        dBax_BclXL_cxdt = b1 * BclXL * Bax - u1 * Bax_BclXL_cx - g16 * Bax_BclXL_cx
        dBad_0dt =  d9 * Bad_p + u2 * (BclXL_tot - (BclXL + Bax_BclXL_cx)) - p7 * AKT_p * Bad_0 + d9 * \
                    (Bad_tot - ((BclXL_tot - (BclXL + Bax_BclXL_cx)) + Bad_0 + Bad_p)) - b2 * Bad_0 * BclXL
        dBad_pdt = p7 * AKT_p * Bad_0 + u3 * (Bad_tot - ((BclXL_tot - (BclXL + Bax_BclXL_cx)) + Bad_0 + Bad_p)) +\
                   p7 * AKT_p * (BclXL_tot - (BclXL + Bax_BclXL_cx)) - d9 * Bad_p - b3 * Bad_p * \
                   (Fourteen33_tot - (Bad_tot - ((BclXL_tot - (BclXL + Bax_BclXL_cx)) + Bad_0 + Bad_p)))
        dproCaspdt = s7 - g17 * proCasp - a1 * Bax * proCasp - a2 * (Casp**2) * proCasp
        dCaspdt = -g18 * Casp + a1 * Bax * proCasp + a2 * Casp**2 * proCasp

        return np.array([dBax_mRNAdt, dBaxdt, dBclXLdt, dBax_BclXL_cxdt, dBad_0dt, dBad_pdt, dproCaspdt, dCaspdt])

    # set parameter values
    p53_killer = 5e5    # high p53_killer expression
    AKT_p = 0
    Bad_tot = 6e4
    BclXL_tot = 1e5
    Fourteen33_tot = 2e5
    a1 = 3e-10
    a2 = 1e-12
    q0_bax = 1e-5
    q1_bax = 3e-13
    q2 = 3e-3
    s4 = 0.03
    t4 = 0.1
    s7 = 30
    p7 = 3e-9
    d9 = 3e-5
    b1 = 3e-5
    b2 = 3e-3
    b3 = 3e-3
    u1 = 1e-3
    u2 = 1e-3
    u3 = 1e-3
    g4 = 3e-4
    g9 = 1e-4
    g16 = 1e-4
    g17 = 3e-4
    g18 = g17
    h = 2
    params = [p53_killer, AKT_p, Bad_tot, BclXL_tot, Fourteen33_tot,
              a1, a2, q0_bax, q1_bax, q2, s4, t4, s7, p7, d9, b1,
              b2, b3, u1, u2, u3, g4, g9, g16, g17, g18, h]

    # set initial values
    Bax_mRNA = 0
    Bax = 0
    BclXL = 0
    Bax_BclXL_cx = 0
    Bad_0 = 0
    Bad_p = 0
    proCasp = 0
    Casp = 0
    y_init = np.array([Bax_mRNA, Bax, BclXL, Bax_BclXL_cx, Bad_0, Bad_p, proCasp, Casp])


    ### FIND INITIAL STEADY STATE ###
    print("Determine initial steady state(s)...")
    # set simulation time (end point should be large, so the system is in steady state)
    eq_step = 1e4
    eq_duration = 1e7
    t = np.arange(0, eq_duration+1, eq_step)
    t_range = (0, eq_duration)

    # solve initial value problem (simulate system)
    method = "LSODA"
    eq1 = solve_ivp(apoptosis, t_range, y_init, args=(params,),
                   method=method, t_eval=[eq_duration]).y

    # find 2nd initial steady state
    params[1] = 1e5     # set Akt to 1e5
    eq2 = solve_ivp(apoptosis, t_range, y_init, args=(params,),
                   method=method, t_eval=[eq_duration]).y


    ### CREATE BIFURCATION DIAGRAM ###

    ### 1) Set up model for PyDSTools
    # Map parameter values to names
    p53_killer = dst.Par(5e5, 'p53_killer')
    AKT_p = dst.Par(0, 'AKT_p')
    Bad_tot = dst.Par(6e4, 'Bad_tot')
    BclXL_tot = dst.Par(1e5, 'BclXL_tot')
    Fourteen33_tot = dst.Par(2e5, 'Fourteen33_tot')
    a1 = dst.Par(3e-10, 'a1')
    a2 = dst.Par(1e-12, 'a2')
    q0_bax = dst.Par(1e-5, 'q0_bax')
    q1_bax = dst.Par(3e-13, 'q1_bax')
    q2 = dst.Par(3e-3, 'q2')
    s4 = dst.Par(0.03, 's4')
    t4 = dst.Par(0.1, 't4')
    s7 = dst.Par(30, 's7')
    p7 = dst.Par(3e-9, 'p7')
    d9 = dst.Par(3e-5, 'd9')
    b1 = dst.Par(3e-5, 'b1')
    b2 = dst.Par(3e-3, 'b2')
    b3 = dst.Par(3e-3, 'b3')
    u1 = dst.Par(1e-3, 'u1')
    u2 = dst.Par(1e-3, 'u2')
    u3 = dst.Par(1e-3, 'u3')
    g4 = dst.Par(3e-4, 'g4')
    g9 = dst.Par(1e-4, 'g9')
    g16 = dst.Par(1e-4, 'g16')
    g17 = dst.Par(3e-4, 'g17')
    g18 = dst.Par(3e-4, 'g18')
    h = dst.Par(2, 'h')

    # define variables
    Bax_mRNA = dst.Var('Bax_mRNA')
    Bax = dst.Var('Bax')
    BclXL = dst.Var('BclXL')
    Bax_BclXL_cx = dst.Var('Bax_BclXL_cx')
    Bad_0 = dst.Var('Bad_0')
    Bad_p = dst.Var('Bad_p')
    proCasp = dst.Var('proCasp')
    Casp = dst.Var('Casp')

    # define reactions
    Bax_mRNA_rhs = s4 * ((q0_bax + q1_bax * p53_killer ** h) / (q2 + q0_bax + q1_bax * p53_killer ** h)) - g4 * Bax_mRNA
    Bax_rhs = t4 * Bax_mRNA + u1 * Bax_BclXL_cx - b1 * BclXL * Bax - g9 * Bax
    BclXL_rhs = u2 * (BclXL_tot - (BclXL + Bax_BclXL_cx)) + u1 * Bax_BclXL_cx + g16 * Bax_BclXL_cx + p7 * AKT_p * \
                (BclXL_tot - (BclXL + Bax_BclXL_cx)) - b2 * BclXL * Bad_0 - b1 * BclXL * Bax
    Bax_BclXL_cx_rhs = b1 * BclXL * Bax - u1 * Bax_BclXL_cx - g16 * Bax_BclXL_cx
    Bad_0_rhs = d9 * Bad_p + u2 * (BclXL_tot - (BclXL + Bax_BclXL_cx)) - p7 * AKT_p * Bad_0 + d9 * \
                (Bad_tot - ((BclXL_tot - (BclXL + Bax_BclXL_cx)) + Bad_0 + Bad_p)) - b2 * Bad_0 * BclXL
    Bad_p_rhs = p7 * AKT_p * Bad_0 + u3 * (Bad_tot - ((BclXL_tot - (BclXL + Bax_BclXL_cx)) + Bad_0 + Bad_p)) + \
                p7 * AKT_p * (BclXL_tot - (BclXL + Bax_BclXL_cx)) - d9 * Bad_p - b3 * Bad_p * \
                (Fourteen33_tot - (Bad_tot - ((BclXL_tot - (BclXL + Bax_BclXL_cx)) + Bad_0 + Bad_p)))
    proCasp_rhs = s7 - g17 * proCasp - a1 * Bax * proCasp - a2 * (Casp ** 2) * proCasp
    Casp_rhs = -g18 * Casp + a1 * Bax * proCasp + a2 * Casp ** 2 * proCasp

    # Create model
    DSargs = dst.args(name='apoptosis')                 # model name
    DSargs.pars = [                                     # model parameters
        p53_killer, AKT_p, Bad_tot, BclXL_tot,
        Fourteen33_tot, a1, a2, q0_bax, q1_bax,
        q2, s4, t4, s7, p7, d9, b1, b2, b3, u1,
        u2, u3, g4, g9, g16, g17, g18, h
    ]
    DSargs.varspecs = dst.args(                         # ODEs
        Bax_mRNA=Bax_mRNA_rhs,
        Bax=Bax_rhs,
        BclXL=BclXL_rhs,
        Bax_BclXL_cx=Bax_BclXL_cx_rhs,
        Bad_0=Bad_0_rhs,
        Bad_p=Bad_p_rhs,
        proCasp=proCasp_rhs,
        Casp=Casp_rhs
    )
    DSargs.ics = dst.args(                              # initial state
        Bax_mRNA=eq1[0],
        Bax=eq1[1],
        BclXL=eq1[2],
        Bax_BclXL_cx=eq1[3],
        Bad_0=eq1[4],
        Bad_p=eq1[5],
        proCasp=eq1[6],
        Casp=eq1[7]
    )

    # Create ODE System for PyDSTools
    ode = dst.Generator.Vode_ODEsystem(DSargs)


    ### 2) Numerical continuation
    # create empty figure
    fig = plt.figure()

    # define numerical continuation problem
    PC = dst.ContClass(ode)
    PCargs = dst.args(name='EQ1', type='EP-C')          # name and type (equilibrium point curve)
    PCargs.freepars = ['AKT_p']                         # set free parameter
    PCargs.MaxNumPoints = 580
    PCargs.MaxStepSize = 250
    PCargs.MinStepSize = 1
    PCargs.StepSize = 25
    PCargs.LocBifPoints = 'LP'                          # detect limit points (saddle-node)
    PCargs.SaveEigen = True                             # compute and show stability

    # compute the first equilibrium point curve
    print("Numerical Continuation 1...")
    PC.newCurve(PCargs)
    PC['EQ1'].backward()

    # draw first bifurcation curve (p53_arrester vs. CyclinE)
    PC.display(['AKT_p', 'Casp'], stability=True, figure=fig)

    # Change the ODEs initial states to the second steady state
    ode.set(ics={
        'Bax_mRNA': eq2[0],
        'Bax': eq2[1],
        'BclXL': eq2[2],
        'Bax_BclXL_cx': eq2[3],
        'Bad_0': eq2[4],
        'Bad_p': eq2[5],
        'proCasp': eq2[6],
        'Casp': eq2[7]
    })
    # change the initial value of the free parameter to 1e5 (corresponds to the steady state above)
    ode.set(pars={'AKT_p': 1e5})

    # define numerical continuation problem
    PC = dst.ContClass(ode)
    PCargs = dst.args(name='EQ1', type='EP-C')
    PCargs.freepars = ['AKT_p']
    PCargs.MaxNumPoints = 465
    PCargs.MaxStepSize = 250
    PCargs.MinStepSize = 1
    PCargs.StepSize = 25
    PCargs.LocBifPoints = 'LP'
    PCargs.SaveEigen = True

    # compute the second equilibrium point curve
    print("Numerical Continuation 2...")
    PC.newCurve(PCargs)
    PC['EQ1'].forward()

    # draw the second equilibrium point curve
    PC.display(['AKT_p', 'Casp'], stability=True, figure=fig)

    # edit figure
    plt.xlabel(r'$\mathregular{Akt_P}$')
    plt.ylabel('Caspase')
    plt.title(r'$\mathregular{p53_{KILLER}=5\cdot 10^5}$', fontsize=15)
    plt.yscale('log')
    plt.ylim([2e-2, 3e5])
    plt.tick_params(labelsize=10)
    fig.tight_layout()
    plt.show()
    if save:
        fig.savefig('bifurcation_apoptosis_Akt.png', dpi=300)

if __name__ == "__main__":
    main()