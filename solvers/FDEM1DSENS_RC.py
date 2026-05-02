###########################################################################
## --------------------------------------------------------------------- ##
#            SENSITIVITY DISTRIBUTION (REFLECTION COEFFICIENT)            #
# ----------------------------------------------------------------------- ##
###########################################################################
#
#  [SENS_IP, SENS_QP, Error] = FDEM1DSENS_RC(S, M, par)
#
#  Use:
#  Calculates the sensitivity distribution of a given layered soil medium
#  and loop-loop configuration towards a certain physical property using
#  the brute-force or perturbation method. Typical characteristics of the
#  soil medium are stored in the Model structure (M) while the sensor
#  characteristics are stored in the Sensor structure (S).
#
#  Input:
#  S (dict)            Sensor characteristics
#  M (dict)            Model characteristics
#  par                 Sensitivity parameter ('con', 'sus', 'perm')
#
#  Output:
#  SENS_IP             IP sensitivity
#  SENS_QP             QP sensitivity
#  Error (dict)        Estimated max. error (.IP and .QP)
#
#  Created by Daan Hanssens
#  UGent, Belgium
#  January 2017
#
#  Cite:
#  Hanssens, D., Delefortrie, S., De Pue, J., Van Meirvenne, M.,
#  and P. De Smedt. Frequency-Domain Electromagnetic Forward and
#  Sensitivity Modeling: Practical Aspects of modeling a Magnetic Dipole
#  in a Multilayered Half-Space. IEEE Geoscience and Remote Sensing
#  Magazine, 7(1), 74-85

import numpy as np
from solvers.FDEM1DFWD_RC import FDEM1DFWD_RC


def FDEM1DSENS_RC(S, M, par):

    #
    # Store original profile
    #

    op = M[par].copy()

    #
    # Calculate partial derivatives
    #

    n = len(M[par])
    pert           = np.zeros(n)
    FWD_IP_alt_p   = np.zeros(n)
    FWD_QP_alt_p   = np.zeros(n)
    FWD_IP_alt_n   = np.zeros(n)
    FWD_QP_alt_n   = np.zeros(n)
    FWD_IP_ori     = 0.0
    FWD_QP_ori     = 0.0

    # Loop Model layers
    for i in range(n):

        #
        # Get original response
        #

        if i == 0:
            FWD_IP_ori, FWD_QP_ori = FDEM1DFWD_RC(S, M)

        #
        # Get altered response (forward)
        #

        M[par] = op.copy()                                                 # Get original profile
        pert[i] = op[i] * 0.01                                             # Get relative perturbation (1%)
        M[par][i] = op[i] + pert[i]
        FWD_IP_alt_p[i], FWD_QP_alt_p[i] = FDEM1DFWD_RC(S, M)

        #
        # Get altered response (backward)
        #

        M[par][i] = op[i] - pert[i]
        FWD_IP_alt_n[i], FWD_QP_alt_n[i] = FDEM1DFWD_RC(S, M)

    #
    # First derivative (Output)
    #

    SENS_QP = (FWD_QP_alt_p - FWD_QP_ori) / pert
    SENS_IP = (FWD_IP_alt_p - FWD_IP_ori) / pert

    #
    # Second derivative
    #

    SENS_QP_pert_sd = (FWD_QP_alt_p - 2*FWD_QP_ori + FWD_QP_alt_n) / pert**2
    SENS_IP_pert_sd = (FWD_IP_alt_p - 2*FWD_IP_ori + FWD_IP_alt_n) / pert**2

    #
    # Estimate maximum error (Output)
    #

    Error = {}
    Error['QP'] = np.max(SENS_QP_pert_sd * pert/2)
    Error['IP'] = np.max(SENS_IP_pert_sd * pert/2)

    return SENS_IP, SENS_QP, Error
