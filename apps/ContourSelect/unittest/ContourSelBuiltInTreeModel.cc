#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int predict(float features[5]) {

    int classes[2];
        
    if (features[0] <= 0.385035157204) {
        if (features[0] <= 0.374575078487) {
            if (features[2] <= 0.323027670383) {
                if (features[0] <= 0.350465655327) {
                    if (features[2] <= 0.31005769968) {
                        classes[0] = 5590; 
                        classes[1] = 0; 
                    } else {
                        if (features[2] <= 0.310163080692) {
                            classes[0] = 0; 
                            classes[1] = 1; 
                        } else {
                            classes[0] = 58; 
                            classes[1] = 0; 
                        }
                    }
                } else {
                    if (features[2] <= 0.302373826504) {
                        if (features[1] <= 0.15952372551) {
                            classes[0] = 1; 
                            classes[1] = 2; 
                        } else {
                            classes[0] = 379; 
                            classes[1] = 5; 
                        }
                    } else {
                        if (features[1] <= 0.365494370461) {
                            classes[0] = 6; 
                            classes[1] = 16; 
                        } else {
                            classes[0] = 3; 
                            classes[1] = 0; 
                        }
                    }
                }
            } else {
                if (features[0] <= 0.332325994968) {
                    classes[0] = 3; 
                    classes[1] = 0; 
                } else {
                    if (features[3] <= 0.330413430929) {
                        classes[0] = 0; 
                        classes[1] = 9; 
                    } else {
                        classes[0] = 1; 
                        classes[1] = 0; 
                    }
                }
            }
        } else {
            if (features[2] <= 0.296626478434) {
                if (features[4] <= 0.982703924179) {
                    classes[0] = 0; 
                    classes[1] = 2; 
                } else {
                    if (features[2] <= 0.289597451687) {
                        classes[0] = 85; 
                        classes[1] = 0; 
                    } else {
                        if (features[2] <= 0.290503770113) {
                            classes[0] = 0; 
                            classes[1] = 1; 
                        } else {
                            classes[0] = 15; 
                            classes[1] = 1; 
                        }
                    }
                }
            } else {
                if (features[3] <= 0.385128855705) {
                    if (features[0] <= 0.374825209379) {
                        if (features[4] <= 0.999873876572) {
                            classes[0] = 2; 
                            classes[1] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 1; 
                        }
                    } else {
                        if (features[0] <= 0.38475716114) {
                            classes[0] = 2; 
                            classes[1] = 46; 
                        } else {
                            classes[0] = 1; 
                            classes[1] = 0; 
                        }
                    }
                } else {
                    classes[0] = 7; 
                    classes[1] = 0; 
                }
            }
        }
    } else {
        if (features[0] <= 0.408606171608) {
            if (features[2] <= 0.290679395199) {
                if (features[1] <= 0.235381826758) {
                    classes[0] = 0; 
                    classes[1] = 27; 
                } else {
                    if (features[2] <= 0.262484431267) {
                        if (features[3] <= 0.172657966614) {
                            classes[0] = 2; 
                            classes[1] = 0; 
                        } else {
                            classes[0] = 1; 
                            classes[1] = 8; 
                        }
                    } else {
                        if (features[2] <= 0.284261345863) {
                            classes[0] = 106; 
                            classes[1] = 1; 
                        } else {
                            classes[0] = 14; 
                            classes[1] = 5; 
                        }
                    }
                }
            } else {
                if (features[2] <= 0.352234899998) {
                    if (features[3] <= 0.357776075602) {
                        if (features[2] <= 0.298841297626) {
                            classes[0] = 3; 
                            classes[1] = 53; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 260; 
                        }
                    } else {
                        if (features[3] <= 0.359698951244) {
                            classes[0] = 1; 
                            classes[1] = 0; 
                        } else {
                            classes[0] = 2; 
                            classes[1] = 18; 
                        }
                    }
                } else {
                    classes[0] = 1; 
                    classes[1] = 0; 
                }
            }
        } else {
            if (features[0] <= 0.428229808807) {
                if (features[2] <= 0.281512588263) {
                    if (features[1] <= 0.330040276051) {
                        classes[0] = 0; 
                        classes[1] = 13; 
                    } else {
                        classes[0] = 27; 
                        classes[1] = 0; 
                    }
                } else {
                    if (features[2] <= 0.28660851717) {
                        if (features[1] <= 0.362408041954) {
                            classes[0] = 1; 
                            classes[1] = 16; 
                        } else {
                            classes[0] = 4; 
                            classes[1] = 1; 
                        }
                    } else {
                        if (features[2] <= 0.357339262962) {
                            classes[0] = 0; 
                            classes[1] = 581; 
                        } else {
                            classes[0] = 1; 
                            classes[1] = 0; 
                        }
                    }
                }
            } else {
                if (features[0] <= 0.431252300739) {
                    if (features[0] <= 0.431168794632) {
                        if (features[1] <= 0.552788555622) {
                            classes[0] = 0; 
                            classes[1] = 128; 
                        } else {
                            classes[0] = 1; 
                            classes[1] = 0; 
                        }
                    } else {
                        if (features[4] <= 0.996213316917) {
                            classes[0] = 0; 
                            classes[1] = 1; 
                        } else {
                            classes[0] = 2; 
                            classes[1] = 0; 
                        }
                    }
                } else {
                    if (features[4] <= 0.843202233315) {
                        if (features[4] <= 0.837617993355) {
                            classes[0] = 0; 
                            classes[1] = 37; 
                        } else {
                            classes[0] = 1; 
                            classes[1] = 0; 
                        }
                    } else {
                        if (features[2] <= 0.255064666271) {
                            classes[0] = 1; 
                            classes[1] = 354; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 25634; 
                        }
                    }
                }
            }
        }
    }

    int index = 0;
    for (int i = 0; i < 2; i++) {
        index = classes[i] > classes[index] ? i : index;
    }
    return index;
}

int main(int argc, const char * argv[]) {

    /* Features: */
    double features[argc-1];
    int i;
    for (i = 1; i < argc; i++) {
        features[i-1] = atof(argv[i]);
    }

    /* Prediction: */
    printf("%d", predict(features));
    return 0;

}