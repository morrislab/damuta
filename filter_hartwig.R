load('data/hmf.mut-type.RData')

# take only last 96 columns 
hw <- mut.df.ct[,55:150]

# restyle naming


write.csv(file ='data/hartwig_counts.csv', hw, quote=F)
