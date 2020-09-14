from enum import IntEnum

# UQ_USE_MPI = False

class UQAlg(IntEnum):
    # Undefined
    NONE = 0
    # Sampling algorithms
    RS = 10
    MCMC = 11
    # Gradient-based algorithms
    SNOPT_RML = 20
    LBFGS_RML = 21
    # Derivative-free algorithms
    MADS_RML = 30
    # Ensemble algorithms
    EnKF = 40
    EnS = 41
    ES_MDA = 42
    #DSI
    DSI = 50

# class UQEnAlg(IntEnum):
#     EnKF = 0
#     EnS = 1
#     ES_MDA = 2
#
#
# class UQGradAlg(IntEnum):
#     SNOPT_RML = 0
#     LBFGS_RML = 1
#
#
# class UQDFOAlg(IntEnum):
#     MADS_RML = 0
#
#
# class UQDSIAlg(IntEnum):
#     DSI = 0
#
#
# class UQSampAlg(IntEnum):
#     RS = 0
#     MCMC = 1
