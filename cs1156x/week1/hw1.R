# The Perceptron Learning Algorithm
#
#   In this problem, you will create your own target function f and data set D to see
#   how the Perceptron Learning Algorithm works. Take d = 2 so you can visualize the
#   problem, and assume X = [-1,1] x [-1,1] with uniform probability of picking each
#   x in X.
#
#   In each run, choose a random line in the plane as your target function f (do this by
#   taking two random, uniformly distributed points in [-1,1] x [-1,1] and taking the
#   line passing through them), where one side of the line maps to +1 and the other maps
#   to -1. Choose the inputs x_n of the data set as random points (uniformly in X), and
#   evaluate the target function on each x_n to get the corresponding output y_n.
#
#   7. Take N = 10. Run the Perceptron Learning Algorithm to find g and measure
#      the disagreement between f and g as P[f(x) != g(x)] (the probability that f and
#      g will disagree on their classification of a random point). You can either calcu-
#      late this exactly, or approximate it by generating a sufficiently large separate
#      set of points to estimate it. Repeat the experiment for 1000 runs (as specified
#      above) and take the average. Start the PLA with the weight vector w being all
#      zeros, and at each iteration have the algorithm choose a point randomly from
#      the set of misclassified points.
#
#      How many iterations does it take on average for the PLA to converge for N = 10
#      training points? Pick the value closest to your results (again, closest is the
#      answer that makes the expression |your answer .. given option| closest to 0).
#
#       [a] 1
#       [b] 15
#       [c] 300
#       [d] 5000
#       [e] 10000

# classify.linear <- function(x,w,b,d=2)
# {
#   distance.from.plane = function(z,w,b) { sum(z*w) + b }
#   # print(x); print(w); print(b)
#   if(class(x)=="numeric"){ x = matrix(x,ncol=d) }
#   distances = apply(x, 1, distance.from.plane, w=w, b=b)
#   return(ifelse(distances < 0, -1, +1))
# }
#
#
# perceptron_web = function(x, y, learning.rate=1)
# {
#   w = vector(length = ncol(x)) # Initialize the parameters
#   b = 0
#   k = 0 # Keep track of how many mistakes we make
#   ni = 0
#   R = max(euclidean.norm(x))
#   made.mistake = TRUE # Initialized so we enter the while loop
#
#   while (made.mistake)
#   {
#     made.mistake=FALSE # Presume that everything's OK
#
#     for (i in 1:nrow(x))
#     {
#       yhat = classify.linear(x[i,],w,b,d=2)
#       cat(printf("y=%d : x=%s -> yhat=%d\n",y[i],paste(sprintf("%.3f",x[i,]),collapse=" "),yhat))
#       if (y[i] != yhat) {
#         w <- w + learning.rate * y[i]*x[i,]
#         b <- b + learning.rate * y[i]*R^2
#         k <- k+1
#         made.mistake=TRUE # Doesn't matter if already set to TRUE previously
#       }
#     }
#     ni <- ni + 1
#   }
#   return(list(w=w,b=b,k=k,ni=ni))
# }

euclidean.norm <- function(x)
{
  return( sqrt(x^2) )
}

pla_classify <- function(x,w,b) { return(ifelse(sum(x*w)+b>0, 1,-1)) }

#  x : N x 2 matrix, columns as features, rows as samples
#  y : N vector containing output
pla_run <- function(x, y, d=2, winit=NULL)
{
  w = rep(0,d)
  if(!is.null(winit) && length(winit)==ncol(x)) { w = winit }

  b = 0
  R = max(euclidean.norm(x))

  ws <- matrix(w, ncol=2)
  bs <- c(b)

  niter = 0
  nmistakes = 0
  has_mistake = TRUE
  while(has_mistake)
  {
    has_mistake = FALSE

    for (i in 1:nrow(x))
    {
      yhat = pla_classify(x[i,], w, b)
      # if classified wrong, update the weight
      if(y[i]*yhat < 0)
      {
        w <- w + y[i]*x[i,]
        b <- b + y[i]*R^2
        nmistakes <- nmistakes + 1
        has_mistake = TRUE
        ws <- rbind(ws, w)
        bs <- c(bs, b)
      }
    }
    niter <- niter + 1
  }
  return(list(w=w, b=b, ws=ws, bs=bs, ni=niter, mi=nmistakes))
}



plotpla <- function(l, file=NULL)
{
  if(!is.null(file)){ GDD(file=file, height=800, width=800) }
  x   = l[['x']]
  y   = l[['y']]
  w   = l[['w']]
  b   = l[['b']]
  ws  = l[['ws']]; ws=NULL
  bs  = l[['bs']]
  tf  = l[['tf']]
  tfb = l[['tfb']]
  ni  = l[['ni']]
  ni  = l[['mi']]

  print(w); print(b)
  print(tf); print(tfb)

  plot(x[,1],x[,2], pch=ifelse(y>0,'o','x'), col=ifelse(y>0,'blue','red'), ylim=c(-1,1),xlim=c(-1,1))
  mtext(sprintf("n_iter: %d, w = %s, tf = %s\n",ni,
    paste(sprintf('%.3f',w),collapse=" "),
    paste(sprintf('%.3f',tf),collapse=" ")), side=3,col='brown')
  abline(v=0,lty=2,col='gray')
  abline(h=0,lty=2,col='gray')
  # lines(c(-5,5), c((-w[length(w)]+5*w[1])/w[2],(-w[length(w)]-5*w[1])/w[2]))
  # lines(c(-1,1),c( (-w[2] + 1*w[1])/w[2], (-w[2] - 1*w[1])/w[2]))

  # decision line is perpendicular to the w vector, which has negative reciprocal slope
  lines(c(-1,1), c(tf[1]/tf[2], -tf[1]/tf[2]) + tfb, lty=1, col='black')

  if(!is.null(ws) && dim(ws)[1]>0){
    print(ws); print(bs)
    cls <- colors()[grep("green",colors())]
    for (i in 1:nrow(ws)) {
      w1 = ws[i,]
      b1 = bs[i]
      lines(c(-1,1), c(w1[1]/w1[2], -w1[1]/w1[2]) + b1, lty=3, col=cls[i])
    }
  }
  lines(c(-1,1), c(w[1]/w[2], -w[1]/w[2]) + b, lty=3, lwd=2, col='darkgreen')

  legend("bottomleft", legend=c("target_func","PLA hypothesis"), fill=c('black','green'))
  if(!is.null(file)){ dev.off() }
}


randpoint <- function(d=2) { runif(d, min=-1,max=1) }
createline <- function()
{
  p1 = randpoint(2)  # p1 = (x1,x2)
  p2 = randpoint(2)  # p2 = (x1,x2)
  tf = p2 - p1
  m  = tf[2]/tf[1]
  tfb = p1[2] - m * p1[1]
  return(list(m=m,tf=tf,tfb=tfb))
}

calcprob <- function(l)
{
  w   = l[['w']]
  b   = l[['b']]
  tf  = l[['tf']]
  tfb = l[['tfb']]

  N = 200
  x = matrix(runif(2*N,min=-1,max=1), ncol=2)
  y    = apply(x, 1, function(x1,tf,tfb) { ifelse((sum(x1*tf)+tfb)>0, 1, -1) }, tf=tf,tfb=tfb)
  yhat = apply(x, 1, function(x1,tf,tfb) { ifelse((sum(x1*tf)+tfb)>0, 1, -1) }, tf=w,tfb=b)

  # number of times outcome is different between target func and hypothesis
  p = length(which(y*yhat<0)) / N
  return(p)
}

test <- function(N=10)
{
  line = createline()
  tf  = line[['tf']]
  tfb = line[['tfb']]

  x = matrix(runif(2*N,min=-1,max=1), ncol=2)
  y = apply(x, 1, function(x1,tf,tfb) { ifelse((sum(x1*tf)+tfb)>0, 1, -1) }, tf=tf,tfb=tfb)

  # l = perceptron(x, y)
  l = pla_run(x, y)

  l[['tf']]  = tf
  l[['tfb']] = tfb
  l[['x']]   = x
  l[['y']]   = y
  return(l)
}

testavg <- function(N=10, iter=1000)
{
  ns = c()
  pb = c()
  for(i in 1:iter)
  {
    line = createline()
    tf  = line[['tf']]
    tfb = line[['tfb']]
    x = matrix(runif(2*N,min=-1,max=1), ncol=2)
    y = apply(x, 1, function(x1,tf,tfb) { return(ifelse(sum(x1*tf)+tfb>0, 1.0, -1.0)) }, tf=tf,tfb=tfb)
    l = pla_run(x, y)
    p = calcprob(l)
    ns <- c(ns, l[['mi']])
    pb <- c(pb, p)
  }
  return(list(niter=mean(ns), prob=mean(pb)))
}


