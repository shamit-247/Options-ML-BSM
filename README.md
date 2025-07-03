# Options-ML-BSM

In computational finance and risk management, several numerical methods (e.g.,
finite differences, fourier methods, and Monte Carlo simulation) are commonly used
for the valuation of financial derivatives.

**The Black-Scholes** formula is probably one of the most widely cited and used models
in derivative pricing. Numerous variations and extensions of this formula are used to
price many kinds of financial derivatives. However, the model is based on several
assumptions. It assumes a specific form of movement for the derivative price, namely
a **Geometric Brownian Motion (GBM)**. It also assumes a conditional payment at
maturity of the option and economic constraints, such as no-arbitrage. Several other
derivative pricing models have similarly impractical model assumptions

Another aspect of the many traditional derivative pricing models is **model calibration**, which is typically done not by historical asset prices but by means of derivative
prices (i.e., by matching the market prices of heavily traded options to the derivative
prices from the mathematical model). In the process of model calibration, thousands
of derivative prices need to be determined in order to fit the parameters of the model,
and the overall process is time consuming. Efficient numerical computation is
increasingly important in financial risk management, especially when we deal with
real-time risk management (e.g., high frequency trading). However, due to the
requirement of a highly efficient computation, certain high-quality asset models and
methodologies are discarded during model calibration of traditional derivative pricing models.

Machine learning can potentially be used to tackle these drawbacks related to imprac‐
tical model assumptions and inefficient model calibration. Machine learning algo‐
rithms have the ability to tackle more nuances with very few theoretical assumptions
and can be effectively used for derivative pricing, even in a world with frictions. With
the advancements in hardware, we can train machine learning models on high per‐
formance CPUs, GPUs, and other specialized hardware to achieve a speed increase of
several orders of magnitude as compared to the traditional derivative pricing models.
Additionally, market data is plentiful, so it is possible to train a machine learning
algorithm to learn the function that is collectively generating derivative prices in the
market. Machine learning models can capture subtle nonlinearities in the data that
are not obtainable through other statistical approaches.

In this dossier, we look at **derivative pricing from a machine learning** standpoint
and use a supervised regression–based model to price an option from simulated data.

**The main idea here is to come up with a machine learning framework for derivative
pricing. Achieving a machine learning model with high accuracy would mean that we can leverage the efficient numerical calculation of machine learning for derivative
pricing with fewer underlying model assumptions.**


In the supervised regression framework used for this case study, the derivative pricing problem is defined in the regression framework, where the predicted variable is the pricing of the option, and the predictor variables are the market data that are used as inputs to the Black-Scholes option pricing model

Options have been used in finance as means to hedge risk in a nonlinear manner. They are are also used by speculators in order to take leveraged bets in the financial markets. Historically, people have used the Black Scholes formula.


#### The Black Scholes formula


$$ Se^{-q \tau}\Phi(d_1) - e^{-r \tau} K\Phi(d_2) \, $$

With
$$ d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)\tau}{\sigma\sqrt{\tau}} $$

and
$$ d_2 = \frac{\ln(S/K) + (r - q - \sigma^2/2)\tau}{\sigma\sqrt{\tau}} = d_1 - \sigma\sqrt{\tau} $$

Where we have; Stock price $S$; Strike price $K$; Risk-free rate $r$; Annual dividend yield $q$; Time to maturity $\tau = T-t$ (represented as a unit-less fraction of one year); Volatility $\sigma$

In order to make the logic simpler, we define Moneyness as $M = K/S$ and look at the prices in terms of per unit of current stock price. We also set $q$ as $0$

This simplifes the formula down to the following
$$ e^{-q \tau}\Phi\left( \frac{- \ln(M) + (r+ \sigma^2/2 )\tau}{\sigma\sqrt{\tau}}\right) - e^{-r \tau} M\Phi\left( \frac{- \ln(M) + (r - \sigma^2/2)\tau}{\sigma\sqrt{\tau}} \right) \, $$


Now as we consider here q=0, Therefore 




$$ \Phi\left( \frac{- \ln(M) + (r+ \sigma^2/2 )\tau}{\sigma\sqrt{\tau}}\right) - e^{-r \tau} M\Phi\left( \frac{- \ln(M) + (r - \sigma^2/2)\tau}{\sigma\sqrt{\tau}} \right) \, $$


where Cummulative Normal Distribution for the expression is defined as :

$$ N_D1 = \Phi\left( \frac{- \ln(M) + (r+ \sigma^2/2 )\tau}{\sigma\sqrt{\tau}}\right) $$

and 

$$ N_D2 = \Phi\left( \frac{- \ln(M) + (r- \sigma^2/2 )\tau}{\sigma\sqrt{\tau}}\right) $$




#### Volatility or Vol Surface

In the options market, there isn't a single value of volatility which gives us the correct price. We often find the volatility such that the output matches the price
Simulation

Here, we assume the the sturcture of the vol surface and generate the data. In practice, we would source the data from a data vendor.

We use the following function to generate the option volatility surface :
$$ \sigma(M, \tau) = \sigma_0 + \alpha\tau + \beta (M - 1)^2$$

