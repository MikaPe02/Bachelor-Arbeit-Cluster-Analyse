
# STICHPROBE

  Variable                            N        MW        SD       Min       Max
  ------------------------------------------------------------------------
  Körpergröße [m]                    60     1.772     0.090     1.570     1.940
  Gewicht [kg]                       60    69.883    10.676    46.700    91.900
  Beinlänge [m]                      60     0.908     0.054     0.772     1.000
  Speed [m/s]                        60     3.554     0.619     2.646     5.190
  SF bei km 1.0 [Hz]                 60     2.860     0.147     2.616     3.361
  Duty Factor (DF)                   60     0.686     0.077     0.491     0.850
  SF_norm                            60     0.869     0.040     0.746     0.972
---


# VORANALYSE: SPEED-KORRELATION (km 1.0, N=60)

  Variable             r        R²           p  Interpretation
  -------------------------------------------------------
  DF              -0.749     0.561      0.0000  Stark
  SF_norm          0.454     0.206      0.0003  Moderat
---


# STIL-CLUSTERING — Ward hierarchisch, k=4, Rohdaten (DF + SF_norm, km 1.0)


## Cluster-Größen

  Cluster 1: N = 21
  Cluster 2: N = 15
  Cluster 3: N = 16
  Cluster 4: N = 8

## Clustering-Metriken (Ward, alle k)

     k    Silhouette    Davies-Bouldin   Calinski-Harabasz  best
  ------------------------------------------------------------
     2         0.306             0.984                24.7
     3         0.301             1.118                30.6
     4         0.337             0.969                31.9  <- GEWÄHLT
     5         0.370             0.855                38.2
     6         0.352             0.766                39.0
     7         0.374             0.785                39.7
     8         0.384             0.767                41.1  <- best

## Silhouette pro Cluster (Stil)

  Cluster             N        MW        SD       Min       Max   Grenzf.<0.2
  ----------------------------------------------------------------------
  Cluster 1          21     0.107     0.172    -0.233     0.379            14
  Cluster 2          15     0.430     0.174    -0.007     0.625             2
  Cluster 3          16     0.489     0.090     0.368     0.648             0
  Cluster 4           8     0.460     0.142     0.211     0.612             0
  Gesamt             60     0.337     0.226    -0.233     0.648            16

## Deskriptive Statistik pro Cluster (MW ± SD)

  Variable                                       Cluster 1               Cluster 2               Cluster 3               Cluster 4
  --------------------------------------------------------------------------------------------------------------------------------
  Duty Factor (DF)                       0.682 ± 0.054          0.590 ± 0.035          0.767 ± 0.036          0.713 ± 0.032   
  SF norm                                0.836 ± 0.036          0.888 ± 0.024          0.867 ± 0.016          0.923 ± 0.025   
  Speed ms                               3.308 ± 0.427          4.368 ± 0.424          3.128 ± 0.361          3.526 ± 0.225   
  Körpergröße m                          1.753 ± 0.071          1.783 ± 0.086          1.780 ± 0.112          1.788 ± 0.102   
  Gewicht kg                            68.619 ± 9.764         66.667 ± 10.493        73.556 ± 11.064        71.887 ± 11.968  
  Beinlänge m                            0.897 ± 0.053          0.910 ± 0.053          0.919 ± 0.055          0.911 ± 0.057   
  SF bei km 1.0 Hz                       2.768 ± 0.101          2.921 ± 0.131          2.836 ± 0.110          3.036 ± 0.149   
  DF start                               0.682 ± 0.054          0.590 ± 0.035          0.767 ± 0.036          0.713 ± 0.032   
  DF end                                 0.684 ± 0.055          0.596 ± 0.039          0.764 ± 0.043          0.706 ± 0.030   
  Delta DF                               0.003 ± 0.016          0.006 ± 0.013         -0.003 ± 0.017         -0.007 ± 0.021   
  Slope DF                               0.000 ± 0.002          0.001 ± 0.002          0.000 ± 0.002         -0.001 ± 0.002   
  Slope DF early                         0.001 ± 0.004          0.001 ± 0.002          0.001 ± 0.002          0.001 ± 0.003   
  Slope DF late                         -0.000 ± 0.002          0.000 ± 0.002         -0.002 ± 0.003         -0.003 ± 0.002   
  SF start                               0.836 ± 0.036          0.888 ± 0.024          0.867 ± 0.016          0.923 ± 0.025   
  SF end                                 0.826 ± 0.036          0.883 ± 0.033          0.849 ± 0.027          0.900 ± 0.043   
  Delta SF                              -0.010 ± 0.015         -0.005 ± 0.016         -0.018 ± 0.020         -0.024 ± 0.024   
  Slope SF                              -0.001 ± 0.002         -0.000 ± 0.002         -0.002 ± 0.002         -0.003 ± 0.002   
  Slope SF early                        -0.002 ± 0.003         -0.001 ± 0.002         -0.002 ± 0.004         -0.002 ± 0.004   
  Slope SF late                         -0.001 ± 0.002         -0.000 ± 0.002         -0.002 ± 0.002         -0.002 ± 0.004   

## ANOVA-Ergebnisse Stil-Clustering

  Kategorie        Feature                         Test               F         p    eta2  sig
  -------------------------------------------------------------------------------------
  Biomechanik      DF                              ANOVA         45.837    0.0000   0.711  ***
  Biomechanik      SF_norm                         ANOVA         22.863    0.0000   0.550  ***
  Biomechanik      speed_ms                        ANOVA         31.155    0.0000   0.625  ***
  Biomechanik      body_height_m                   ANOVA          0.525    0.6670   0.027  n.s.
  Biomechanik      weight_kg                       ANOVA          1.297    0.2846   0.065  n.s.
  Biomechanik      leg_length_m                    ANOVA          0.548    0.6513   0.029  n.s.
  Fatigue          DF_start                        ANOVA         45.837    0.0000   0.711  ***
  Fatigue          DF_end                          ANOVA         36.071    0.0000   0.659  ***
  Fatigue          Delta_DF                        ANOVA          1.481    0.2297   0.073  n.s.
  Fatigue          Slope_DF                        ANOVA          1.897    0.1405   0.092  n.s.
  Fatigue          Slope_DF_early                  ANOVA          0.102    0.9584   0.005  n.s.
  Fatigue          Slope_DF_late                   ANOVA          5.304    0.0027   0.221  **
  Fatigue          SF_start                        ANOVA         22.863    0.0000   0.550  ***
  Fatigue          SF_end                          ANOVA         13.202    0.0000   0.414  ***
  Fatigue          Delta_SF                        ANOVA          2.444    0.0735   0.116  n.s.
  Fatigue          Slope_SF                        ANOVA          3.316    0.0263   0.151  *
  Fatigue          Slope_SF_early                  ANOVA          0.492    0.6889   0.026  n.s.
  Fatigue          Slope_SF_late                   ANOVA          1.512    0.2215   0.075  n.s.

## Cohen's d (alle Cluster-Paare) — Stil-Clustering

  Feature                             C1_vs_C2      C1_vs_C3      C1_vs_C4      C2_vs_C3      C2_vs_C4      C3_vs_C4
  ------------------------------------------------------------------------------------------------------------------
  DF                                     1.947        -1.822        -0.647        -4.989        -3.637         1.542
  SF_norm                               -1.637        -1.054        -2.599         1.049        -1.434        -2.945
  speed_ms                              -2.490         0.452        -0.566         3.163         2.280        -1.232
  body_height_m                         -0.397        -0.306        -0.444         0.030        -0.052        -0.071
  weight_kg                              0.194        -0.477        -0.315        -0.638        -0.474         0.147
  leg_length_m                          -0.244        -0.419        -0.261        -0.178        -0.021         0.153
  DF_start                               1.947        -1.822        -0.647        -4.989        -3.637         1.542
  DF_end                                 1.804        -1.599        -0.440        -4.101        -3.030         1.482
  Delta_DF                              -0.201         0.359         0.562         0.583         0.791         0.212
  Slope_DF                              -0.256         0.123         0.750         0.368         0.934         0.603
  Slope_DF_early                        -0.058        -0.141         0.037        -0.135         0.136         0.229
  Slope_DF_late                         -0.289         0.896         1.045         1.085         1.235         0.042
  SF_start                              -1.637        -1.054        -2.599         1.049        -1.434        -2.945
  SF_end                                -1.649        -0.706        -1.954         1.128        -0.455        -1.529
  Delta_SF                              -0.336         0.437         0.742         0.711         0.986         0.262
  Slope_SF                              -0.424         0.327         0.939         0.700         1.338         0.517
  Slope_SF_early                        -0.239         0.222         0.141         0.425         0.362        -0.071
  Slope_SF_late                         -0.281         0.337         0.498         0.646         0.723         0.226

## Tukey HSD — signifikante Vergleiche (Stil)

  Kategorie        Feature                      C_A    C_B      Diff     p_adj    CI_low   CI_high
  ------------------------------------------------------------------------------------------
  Biomechanik      DF                         Cluster 1  Cluster 2    -0.091    0.0000    -0.129    -0.053
  Biomechanik      DF                         Cluster 1  Cluster 3     0.086    0.0000     0.048     0.123
  Biomechanik      DF                         Cluster 2  Cluster 3     0.177    0.0000     0.136     0.217
  Biomechanik      DF                         Cluster 2  Cluster 4     0.123    0.0000     0.073     0.172
  Biomechanik      DF                         Cluster 3  Cluster 4    -0.054    0.0245    -0.103    -0.005
  Biomechanik      SF_norm                    Cluster 1  Cluster 2     0.052    0.0000     0.027     0.076
  Biomechanik      SF_norm                    Cluster 1  Cluster 3     0.031    0.0075     0.006     0.055
  Biomechanik      SF_norm                    Cluster 1  Cluster 4     0.087    0.0000     0.057     0.117
  Biomechanik      SF_norm                    Cluster 2  Cluster 4     0.035    0.0244     0.003     0.067
  Biomechanik      SF_norm                    Cluster 3  Cluster 4     0.057    0.0001     0.025     0.088
  Biomechanik      speed_ms                   Cluster 1  Cluster 2     1.060    0.0000     0.712     1.408
  Biomechanik      speed_ms                   Cluster 2  Cluster 3    -1.241    0.0000    -1.611    -0.871
  Biomechanik      speed_ms                   Cluster 2  Cluster 4    -0.842    0.0000    -1.293    -0.391
  Fatigue          DF_start                   Cluster 1  Cluster 2    -0.091    0.0000    -0.129    -0.053
  Fatigue          DF_start                   Cluster 1  Cluster 3     0.086    0.0000     0.048     0.123
  Fatigue          DF_start                   Cluster 2  Cluster 3     0.177    0.0000     0.136     0.217
  Fatigue          DF_start                   Cluster 2  Cluster 4     0.123    0.0000     0.073     0.172
  Fatigue          DF_start                   Cluster 3  Cluster 4    -0.054    0.0245    -0.103    -0.005
  Fatigue          DF_end                     Cluster 1  Cluster 2    -0.088    0.0000    -0.129    -0.048
  Fatigue          DF_end                     Cluster 1  Cluster 3     0.080    0.0000     0.040     0.119
  Fatigue          DF_end                     Cluster 2  Cluster 3     0.168    0.0000     0.125     0.211
  Fatigue          DF_end                     Cluster 2  Cluster 4     0.110    0.0000     0.057     0.162
  Fatigue          DF_end                     Cluster 3  Cluster 4    -0.058    0.0228    -0.110    -0.006
  Fatigue          Slope_DF_late              Cluster 1  Cluster 3    -0.002    0.0415    -0.004    -0.000
  Fatigue          Slope_DF_late              Cluster 2  Cluster 3    -0.003    0.0101    -0.005    -0.001
  Fatigue          Slope_DF_late              Cluster 2  Cluster 4    -0.003    0.0363    -0.006    -0.000
  Fatigue          SF_start                   Cluster 1  Cluster 2     0.052    0.0000     0.027     0.076
  Fatigue          SF_start                   Cluster 1  Cluster 3     0.031    0.0075     0.006     0.055
  Fatigue          SF_start                   Cluster 1  Cluster 4     0.087    0.0000     0.057     0.117
  Fatigue          SF_start                   Cluster 2  Cluster 4     0.035    0.0244     0.003     0.067
  Fatigue          SF_start                   Cluster 3  Cluster 4     0.057    0.0001     0.025     0.088
  Fatigue          SF_end                     Cluster 1  Cluster 2     0.057    0.0000     0.026     0.088
  Fatigue          SF_end                     Cluster 1  Cluster 4     0.074    0.0000     0.036     0.111
  Fatigue          SF_end                     Cluster 2  Cluster 3    -0.034    0.0353    -0.067    -0.002
  Fatigue          SF_end                     Cluster 3  Cluster 4     0.051    0.0058     0.012     0.090
  Fatigue          Slope_SF                   Cluster 2  Cluster 4    -0.002    0.0214    -0.004    -0.000

## Chi-Quadrat kategoriale Variablen (Stil)

  Variable                  chi2    df         p  sig
  ------------------------------------------------
  sex                      2.345     3    0.5039  n.s.
  dominant_leg             1.141     3    0.7673  n.s.
---


# FATIGUE-CLUSTERING — Ward hierarchisch, k=3, Fatigue-Features (Delta+Slope DF+SF)


## Cluster-Größen

  Cluster 1: N = 18
  Cluster 2: N = 34
  Cluster 3: N = 8

## Clustering-Metriken (Ward, alle k)

     k    Silhouette    Davies-Bouldin   Calinski-Harabasz  best
  ------------------------------------------------------------
     2         0.390             0.909                42.5
     3         0.396             0.907                45.7  <- GEWÄHLT
     4         0.388             0.857                41.7
     5         0.316             0.816                42.2
     6         0.327             0.855                41.3
     7         0.322             0.848                43.2
     8         0.320             0.911                44.8

## Silhouette pro Cluster (Fatigue)

  Cluster             N        MW        SD       Min       Max   Grenzf.<0.2
  ----------------------------------------------------------------------
  Cluster 1          18     0.308     0.188    -0.166     0.484             5
  Cluster 2          34     0.466     0.113     0.209     0.626             0
  Cluster 3           8     0.298     0.106     0.150     0.437             2
  Gesamt             60     0.396     0.159    -0.166     0.626             7

## Deskriptive Statistik pro Cluster (MW ± SD)

  Variable                                          Cluster 1               Cluster 2               Cluster 3
  -----------------------------------------------------------------------------------------------------------
  DF Start (km 1)                           0.670 ± 0.072          0.686 ± 0.084          0.719 ± 0.049   
  SF norm Start (km 1)                      0.864 ± 0.045          0.871 ± 0.034          0.874 ± 0.053   
  Speed ms                                  3.589 ± 0.673          3.604 ± 0.634          3.262 ± 0.351   
  Koerpergroesse m                          1.737 ± 0.092          1.797 ± 0.081          1.747 ± 0.098   
  Gewicht kg                               66.933 ± 11.136        71.279 ± 9.648         70.588 ± 13.719  
  Beinlaenge m                              0.884 ± 0.062          0.922 ± 0.045          0.900 ± 0.055   
  Delta DF                                  0.015 ± 0.013         -0.000 ± 0.008         -0.030 ± 0.009   
  Slope DF                                  0.002 ± 0.002          0.000 ± 0.001         -0.003 ± 0.001   
  Slope DF early                            0.003 ± 0.002          0.001 ± 0.002         -0.003 ± 0.003   
  Slope DF late                             0.001 ± 0.002         -0.001 ± 0.002         -0.004 ± 0.003   
  Delta SF                                  0.007 ± 0.011         -0.019 ± 0.010         -0.035 ± 0.019   
  Slope SF                                  0.001 ± 0.001         -0.002 ± 0.001         -0.004 ± 0.002   
  Slope SF early                            0.000 ± 0.002         -0.002 ± 0.002         -0.005 ± 0.005   
  Slope SF late                             0.001 ± 0.002         -0.002 ± 0.002         -0.003 ± 0.003   

## ANOVA-Ergebnisse Fatigue-Clustering

  Kategorie        Feature                         Test               F         p    eta2  sig
  -------------------------------------------------------------------------------------
  Biomechanik      DF                              ANOVA          1.116    0.3347   0.038  n.s.
  Biomechanik      SF_norm                         ANOVA          0.230    0.7952   0.008  n.s.
  Biomechanik      speed_ms                        ANOVA          1.032    0.3629   0.035  n.s.
  Biomechanik      body_height_m                   ANOVA          3.228    0.0470   0.102  *
  Biomechanik      weight_kg                       ANOVA          0.995    0.3760   0.034  n.s.
  Biomechanik      leg_length_m                    ANOVA          3.228    0.0470   0.102  *
  Fatigue          Delta_DF                        ANOVA         58.883    0.0000   0.674  ***
  Fatigue          Slope_DF                        ANOVA         34.465    0.0000   0.547  ***
  Fatigue          Slope_DF_early                  ANOVA         15.640    0.0000   0.354  ***
  Fatigue          Slope_DF_late                   ANOVA         15.569    0.0000   0.353  ***
  Fatigue          Delta_SF                        ANOVA         41.722    0.0000   0.594  ***
  Fatigue          Slope_SF                        ANOVA         52.511    0.0000   0.648  ***
  Fatigue          Slope_SF_early                  ANOVA         10.027    0.0002   0.260  ***
  Fatigue          Slope_SF_late                   ANOVA         11.986    0.0000   0.296  ***

## Cohen's d (alle Cluster-Paare) — Fatigue-Clustering

  Feature                             C1_vs_C2      C1_vs_C3      C2_vs_C3
  ------------------------------------------------------------------------
  DF                                    -0.203        -0.741        -0.411
  SF_norm                               -0.172        -0.214        -0.095
  speed_ms                              -0.023         0.548         0.576
  body_height_m                         -0.708        -0.106         0.595
  weight_kg                             -0.427        -0.306         0.066
  leg_length_m                          -0.737        -0.270         0.461
  Delta_DF                               1.527         3.775         3.743
  Slope_DF                               1.246         2.756         2.869
  Slope_DF_early                         0.937         2.287         1.433
  Slope_DF_late                          0.931         2.278         1.410
  Delta_SF                               2.416         2.958         1.290
  Slope_SF                               2.583         3.617         1.372
  Slope_SF_early                         1.132         1.559         0.783
  Slope_SF_late                          1.242         1.619         0.644

## Tukey HSD — signifikante Vergleiche (Fatigue)

  Kategorie        Feature                      C_A    C_B      Diff     p_adj    CI_low   CI_high
  ------------------------------------------------------------------------------------------
  Biomechanik      leg_length_m               Cluster 1  Cluster 2     0.038    0.0397     0.002     0.074
  Fatigue          Delta_DF                   Cluster 1  Cluster 2    -0.015    0.0000    -0.022    -0.008
  Fatigue          Delta_DF                   Cluster 1  Cluster 3    -0.045    0.0000    -0.055    -0.035
  Fatigue          Delta_DF                   Cluster 2  Cluster 3    -0.030    0.0000    -0.039    -0.021
  Fatigue          Slope_DF                   Cluster 1  Cluster 2    -0.001    0.0002    -0.002    -0.001
  Fatigue          Slope_DF                   Cluster 1  Cluster 3    -0.004    0.0000    -0.005    -0.003
  Fatigue          Slope_DF                   Cluster 2  Cluster 3    -0.003    0.0000    -0.004    -0.002
  Fatigue          Slope_DF_early             Cluster 1  Cluster 2    -0.002    0.0066    -0.004    -0.001
  Fatigue          Slope_DF_early             Cluster 1  Cluster 3    -0.006    0.0000    -0.008    -0.003
  Fatigue          Slope_DF_early             Cluster 2  Cluster 3    -0.003    0.0018    -0.006    -0.001
  Fatigue          Slope_DF_late              Cluster 1  Cluster 2    -0.002    0.0116    -0.003    -0.000
  Fatigue          Slope_DF_late              Cluster 1  Cluster 3    -0.005    0.0000    -0.007    -0.003
  Fatigue          Slope_DF_late              Cluster 2  Cluster 3    -0.003    0.0010    -0.005    -0.001
  Fatigue          Delta_SF                   Cluster 1  Cluster 2    -0.026    0.0000    -0.035    -0.018
  Fatigue          Delta_SF                   Cluster 1  Cluster 3    -0.042    0.0000    -0.054    -0.030
  Fatigue          Delta_SF                   Cluster 2  Cluster 3    -0.016    0.0040    -0.028    -0.004
  Fatigue          Slope_SF                   Cluster 1  Cluster 2    -0.003    0.0000    -0.004    -0.002
  Fatigue          Slope_SF                   Cluster 1  Cluster 3    -0.004    0.0000    -0.006    -0.003
  Fatigue          Slope_SF                   Cluster 2  Cluster 3    -0.002    0.0019    -0.003    -0.001
  Fatigue          Slope_SF_early             Cluster 1  Cluster 2    -0.003    0.0059    -0.004    -0.001
  Fatigue          Slope_SF_early             Cluster 1  Cluster 3    -0.005    0.0003    -0.008    -0.002
  Fatigue          Slope_SF_late              Cluster 1  Cluster 2    -0.003    0.0004    -0.004    -0.001
  Fatigue          Slope_SF_late              Cluster 1  Cluster 3    -0.004    0.0002    -0.006    -0.002

## Chi-Quadrat kategoriale Variablen (Fatigue)

  Variable                  chi2    df         p  sig
  ------------------------------------------------
  sex                      1.193     2    0.5508  n.s.
  dominant_leg             6.858     2    0.0324  *
---


# KREUZTABELLE: Laufstil-Cluster vs. Fatigue-Cluster

  chi2=7.706, df=6, p=0.2604, sig=n.s.

Fatigue-Cluster  Cluster 1  Cluster 2  Cluster 3
Stil-Cluster                                    
Cluster 1                7         12          2
Cluster 2                6          9          0
Cluster 3                4          9          3
Cluster 4                1          4          3
---
