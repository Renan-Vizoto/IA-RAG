# Dutch Energy Regression Summary

## Dataset Audit

```text
                       Metrica   Valor
Registros apos limpeza inicial 4055119
              Anos disponiveis      12
                Distribuidoras      19
               Areas de compra      17
                       Cidades    2631
              Tipos de conexao      25
      Cidades em mais de 1 ano    2610
```

## Split Summary

```text
Split    Rows   Share_%                                          Years
Train 1876095 59.296612 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016
  Val  580757 18.355639                                     2017, 2018
 Test  707064 22.347749                                     2019, 2020
```

## Model Comparison

```text
                  Modelo         MAE      MedAE        RMSE       MAPE      WMAPE        R2
Rank                                                                                       
1                XGBoost   39.506673  16.567737   89.081388  17.563344  18.982556  0.743352
2               LightGBM   39.538688  16.372806   90.107310  17.509078  18.997939  0.737406
3          Random Forest   40.300589  16.524594   91.113155  18.739800  19.364025  0.731511
4                  Lasso   45.146504  19.900220  103.596275  21.313249  21.692438  0.652901
5      Linear Regression   45.308357  17.375429  133.149114  27.861270  21.770207  0.426622
6                  Ridge   45.336111  19.474794  102.852606  20.551673  21.783542  0.657867
7                    MLP   50.728806  20.199888  105.401374  20.391986  24.374678  0.640700
8     Baseline (Mediana)  105.254030  67.038780  179.224653  69.027343  50.573495 -0.038867
```
