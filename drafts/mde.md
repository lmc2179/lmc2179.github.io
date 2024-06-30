Simulating the MDE
* Set the sample size, n
* Set the control rate, r
* Set the significance level, a
* Set the power, b
* Set the simulation count, nsim=10000
* Set the possible effect sizes d1, ..., dk
For each effect size d_i:
    * set detected = 0
    * for _ in range(nsim):
        * simulate a control and treatment data set
        * compute the treatment effect CI at sig level alpha
        * check if the CI spans zero. if not, detected += 1
    * the estimate power for d_i is detected / nsim
* Return the smallest d_i such that the observed detection rate >= p