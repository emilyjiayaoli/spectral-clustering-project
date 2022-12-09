import Pkg
Pkg.add("DelimitedFiles")
Pkg.add("StatsBase")

# Loading Data
using DelimitedFiles
filepath = "./music_genre_data.csv"
data = readdlm(filepath, ',')
#/Users/emily/Desktop/21-241\ Linear/

# Data Info: 1000 data points, 199 dimensions of features

data_num = Float64.(data[2:1000,2:198]) # removed the tile rows and columns, turn into float

# Applied normalization
using StatsBase
using Random

X_standard = standardize(ZScoreTransform, data_num, dims=2) #standardizing accross the columns, so dims=2
#X_standard = data_num

# First parameter options: UnitRangeTransform, ZScoreTransform
    # https://juliastats.org/StatsBase.jl/stable/transformations/#Unit-range-normalization-1

X_transposed = transpose(X_standard) # rows are dimensions, columns are datapoints
# X = X_transposed[:, shuffle(1:999)]

X = X_transposed[:, 1:999]

numNeighbors = 15
numPoints = 999
#numClusters = 10 # we figure out what this value is based of the first gap in eigenvalues

using DataFrames
using NearestNeighbors
kdtree = KDTree(X)

idx, dists = knn(kdtree, X, numNeighbors + 1, true)

# Turns vector within vector into matrix
idxMatrix = mapreduce(permutedims, vcat, idx)


# Adjancency list (list of indexes for k nearest neighbors at each index): 
neighbors = idxMatrix[:,2:numNeighbors + 1]


# Creating adjancency matrix
adj = zeros(numPoints, numPoints)

for i in 1:numPoints 
    for j in 1:numNeighbors
      adj[i, neighbors[i, j]] = 1
      adj[neighbors[i, j], i] = 1
    end
  end


degree = zeros(numPoints, numPoints)  

for i in 1:numPoints
    degree[i, i] = sum(adj[i,:])
  end

laplacian = degree - adj


using LinearAlgebra
eigenval, eigenvec = eigen(laplacian)


scatter(collect(1:999), eigenval, xlim=(0, 15), ylim=(0,2.5), xlabel="count", ylabel="eigenvalue", title="\n Dataset #2 - Eigenvalues for Laplacian Matrix",titlefont=font(12), bottom_margin = 10mm, top_margin = 5mm, left_margin = 10mm, right_margin = 10mm)

# Specify number of clusters
numClusters = 10

# Retrieve specified amount of eigenvectors
A = eigenvec[:,2:numClusters + 1]

# Perform k means
using RDatasets, Clustering
km = kmeans(transpose(A), numClusters)


# Plotting results.
using Plots
using Plots.PlotMeasures
s = 2.5
scatter(X[1, :], X[31, :], marker_z=km.assignments, color=:lightrainbow, legend=false, markersize=s)

s = 4
scatter(X[1, :], X[31, :], 
            marker_z=km.assignments, 
            color=:lightrainbow, 
            legend=true, markersize=s, 
            xlabel="tempo (dim 1)", 
            ylabel="rolloff_mean (dim 31)", 
            title="\n Tempo vs Rolloff Mean (original scale)" , 
            titlefont=font(12), 
            bottom_margin = 10mm, 
            top_margin = 5mm, 
            left_margin = 10mm, 
            right_margin = 10mm)


# Access results, clusters to verify and interpret clustering results

@assert nclusters(km) == numClusters # verify the number of clusters

a = assignments(km) # get the assignments of points to clusters
c = counts(km) # get the cluster sizes
M = km.centers # get the cluster centers
