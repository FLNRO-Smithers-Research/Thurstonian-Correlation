#Thurstonian model for ranking data using general mean and covar structures.
#Peter Ott. Nov. 2, 2018. Mark 1. Still room for improving MCMC chains.

#Install JAGS 4.3 (or higher) first: https://sourceforge.net/projects/mcmc-jags/
#install.packages(c("coda")) #rjags needs this installed 
library(rjags)

#----- Data from Maydeau-Olivares, A. 1999. Psychometrika 64(3): 325-340 ----;

fake.ranks.0 <- read.csv("G:\\!Workgrp\\RS\\PKO-consult\\working\\maydeau_olivares.csv")
fake.ranks <- fake.ranks.0[rep(seq_len(nrow(fake.ranks.0)), fake.ranks.0$freq),c(2:5)] 
#starters <- rank(apply(fake.ranks,2,mean))

m.judges <- nrow(fake.ranks)
n.items <- ncol(fake.ranks)  

#--------Defining Thurstonian Model------;
#m is the number of observations (judges)
#n is the number of items being ranked

model.string <- "
data {
 lower <- -1e5
 upper <- 1e5
 eps <- 1e-6
 for (i in 1:m) {
     y.star[i,1:n] <- order(y[i,])
 }
 #Number of correlation params
 n.cor <- n*(n-1)/2
 #Initializing
 for (i in 1:m) {
    for (j in 1:n) {
        nu[i,j] <- 1
    }
 }
}
model {
 #Priors
 mu[n] <- 0
 for (j in 1:(n-1)) {
    #vague prior
    mu[j] ~ dnorm(0,0.001) 
 }
 #Restricting last corr to zero
 rho[n.cor] <- 0
 for (l in 1:(n.cor-1)) {
    #rho[l] ~ dunif(-1, 1)
    #somewhat vague beta prior - uniform is too uninformative. 
    rho[l] ~ dbeta(1.5, 1.5)
 }
 for (j in 1:n) {
    Sigma[j,j] <- 1
    for (k in (j+1):n) {
       Sigma[j,k] <- rho[(j-1)*(n-j/2)+(k-j)]
       Sigma[k,j] <- Sigma[j,k]
    }
 }
 #Likelihood
 for (i in 1:m) {
    #Note vcov suffix means Sigma is NOT a precision matrix 
    u[i,1:n] ~ dmnorm.vcov(mu[1:n], Sigma[1:n,1:n])
    #Since nu is unobserved, dinterval imposes the a-posteriori restriction that u must lie within defined bounds
    for (j in 1:n) {
        #Bounds ensure realizations honor how data is sorted 
        bounds[i,j,1] <- equals(y[i,j],n)*lower + inprod(u[i,], equals(y[i,], y[i,j]+1)) - eps
        bounds[i,j,2] <- equals(y[i,j],1)*upper + inprod(u[i,], equals(y[i,], y[i,j]-1)) + eps
        nu[i,j] ~ dinterval(u[i,j], bounds[i,j,])
    }
 } 
 #Aggregate ranking
 #ystar <- rank(-mu[1:n])
}
"

#Prepare data for Jags input
data <- list(y=fake.ranks, m=m.judges, n=n.items)
n.chains <- 5
jags <- jags.model(textConnection(model.string), data, n.chains=n.chains)

#Number of draws to discard as burn-in
n.burn <- 1000
update(jags, n.burn)

#Optional. Set initial values and seed for reproducibility. Must define both .RNG.name and .Rng.seed 
#mod.inits <- list(mu=starters, .RNG.name="base::Super-Duper", .RNG.seed=19)
n.iter <- 10000
int.thin <- 10 #make at least 10, but 15+ is even better!
n.adapt <- 7500
samp <- coda.samples(jags, c("mu","rho"), n.iter=n.iter, thin=int.thin, n.adapt=n.adapt)
#samp <- coda.samples(jags, c("mu","rho"), inits=mod.inits, n.iter=n.iter, thin=int.thin, n.adapt=n.adapt)

#str(samp) 
#colnames(samp[[1]]) #colnames of 1st chain
rho.ind <- which(grepl("rho|mu", colnames(samp[[1]])))
#--All chains together
summary(samp[,rho.ind])$stat
autocorr(samp[,rho.ind]) 
effectiveSize(samp[,rho.ind])
par(ask=TRUE)
plot(samp[,rho.ind]) 

#--Convergence diagnostics
#Gelman-Rubin Diagnostic
#Look for G-R test of R ~ 1, values above 1.05 indicate lack of convergence.
gelman.diag(samp)
 g <- matrix(NA, nrow=nvar(samp), ncol=2)
 for (v in 1:nvar(samp)) {
    g[v,] <- gelman.diag(samp[,v])$psrf
 }
#Geweke Diagnostic - w/i a single chain, looks at 1st and 2nd half and compares distributions
#The Geweke statistic is N(0,1) under H0, so values beyond -2.5 or 2.5 indicate nonstationarity of chain, and that burn-in is not sufficient.
geweke.diag(samp[,rho.ind])
# raftery.diag(jm1.samp[ ,tmp.ind])
heidel.diag(samp[,rho.ind])

#Combine all chains into matrix
samp.mat <- as.matrix(samp)
dim(samp.mat) #[1] 5000   10
summary(as.mcmc(samp.mat))
HPDinterval(as.mcmc(samp.mat))