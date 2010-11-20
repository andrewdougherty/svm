{-
SVM is an implementation of a support vector machine in the Haskell language.
Copyright (C) 2010  Andrew Dougherty

Send email to: andrewdougherty@me.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
-}

{-# OPTIONS_GHC -XBangPatterns #-}

{- This module performs support vector regression on a set of training points in order to determine
the generating function.  Currently least squares support vector regression is implemented.  The
optimal solution to the Langrangian is found by a conjugate gradient algorithm (CGA).  The CGA finds
the saddle point of the dual of the Lagrangian.-}
module SVM (DataSet (..), SVMSolution (..), KernelFunction (..), SVM (..), LSSVM (..),
            KernelMatrix (..), reciprocalKernelFunction, radialKernelFunction, linearKernelFunction,
            splineKernelFunction, polyKernelFunction, mlpKernelFunction) where
   
   import Data.Array.Unboxed             -- Unboxed arrays are used for better performance.
   import Data.List (foldl')             -- foldl' gives better performance than sum
   
   -- |The type synonym is simply to save some typing.
   type DoubleArray = UArray Int Double

   -- |Each data set is a list of vectors and values which are training points of the form
   -- f(x) = y forall {x,y}.
   data DataSet = DataSet {points::(Array Int [Double]), values::DoubleArray}
   
   -- |The solution contains the dual weights, the support vectors and the bias.
   data SVMSolution = SVMSolution {alpha::DoubleArray, sv::(Array Int [Double]), bias::Double}
   
   -- |The kernel matrix has been implemented as an unboxed array for performance reasons.
   newtype KernelMatrix = KernelMatrix DoubleArray
   
   {- |Every kernel function represents an inner product in feature space.  The parameters are:
     
      * A list of kernel parameters that can be interpreted differently by each kernel function.
     
      * The first point in the inner product.
     
      * The second point in the inner product.-}
   newtype KernelFunction = KernelFunction ([Double] -> [Double] -> [Double] -> Double)
   
   -- Some common kernel functions (these are called many times, so they need to be fast):
   
   -- |The reciprocal kernel is the result of exponential basis functions, exp(-k*(x+a)).  The inner
   -- product is an integral over all k >= 0.
   reciprocalKernelFunction :: [Double] -> [Double] -> [Double] -> Double
   reciprocalKernelFunction (a:as) (x:xs) (y:ys) = (reciprocalKernelFunction as xs ys) / (x + y + 2*a)
   reciprocalKernelFunction _ _ _ = 1
   
   -- |This is the kernel when radial basis functions are used.
   radialKernelFunction :: [Double] -> [Double] -> [Double] -> Double
   radialKernelFunction (a:as) x y = exp $ (cpshelp 0 x y) / a
            where cpshelp !accum (x:xs) (y:ys) = cpshelp (accum + (x-y)**2) xs ys
                  cpshelp !accum _ _ = negate accum
   
   -- |This is a simple dot product between the two data points, corresponding to a featureless space.
   linearKernelFunction :: [Double] -> [Double] -> [Double] -> Double
   linearKernelFunction (a:as) (x:xs) (y:ys) = x * y + linearKernelFunction as xs ys
   linearKernelFunction _ _ _ = 0
   
   splineKernelFunction :: [Double] -> [Double] -> [Double] -> Double
   splineKernelFunction a x y | dp <= 1.0 = (2/3) - dp^2 + (0.5*dp^3)
                              | dp <= 2.0 = (1/6) * (2-dp)^3
                              | otherwise = 0.0
            where dp = linearKernelFunction a x y
   
   polyKernelFunction :: [Double] -> [Double] -> [Double] -> Double
   polyKernelFunction (a0:a1:as) x y = (a0 + linearKernelFunction as x y)**a1
   
   -- |Provides a solution similar to neural net.
   mlpKernelFunction :: [Double] -> [Double] -> [Double] -> Double
   mlpKernelFunction (a0:a1:as) x y = tanh (a0 * linearKernelFunction as x y - a1)
   
   {- |A support vector machine (SVM) can estimate a function based upon some training data.
   Instances of this class need only implement the dual cost and the kernel function.  Default
   implementations are given for finding the SVM solution, for simulating a function and for
   creating a kernel matrix from a set of training points.  All SVMs should return a solution
   which contains a list of the support vectors and their dual weigths.  dcost represents the
   coefficient of the dual cost function.  This term gets added to the diagonal elements of the
   kernel matrix and may be different for each type of SVM. -}
   class SVM a where
      {- |Creates a 'KernelMatrix' from the training points in the 'DataSet'.  If @kf@ is the
      'KernelFunction' then the elements of the kernel matrix are given by @K[i,j] = kf x[i] x[j]@,
      where the @x[i]@ are taken from the training points.  The kernel matrix is symmetric and
      positive semi-definite.Only the bottom half of the kernel matrix is stored.-}
      createKernelMatrix  :: a -> (Array Int [Double]) -> KernelMatrix
      {- |The derivative of the cost function is added to the diagonal elements of the kernel
      matrix.  This places a cost on the norm of the solution, which helps prevent overfitting
      of the training data.-}
      dcost               :: a -> Double
      -- |This function provides access to the 'KernelFunction' used by the 'SVM'.
      evalKernel          :: a -> [Double] -> [Double] -> Double
      {- |This function takes an 'SVMSolution' produced by the 'SVM' passed in, and a list of points
      in the space, and it returns a list of valuues y = f(x), where f is the generating function
      represented by the support vector solution.-}
      simulate            :: a -> SVMSolution -> (Array Int [Double]) -> [Double]
      {- |This function takes a 'DataSet' and feeds it to the 'SVM'.  Then it returns the
      'SVMSolution' which is the support vector solution for the function which generated the points
      in the training set.  The function also takes values for epsilon and the max iterations, which
      are used as stopping criteria in the conjugate gradient algorithm.-}
      solve               :: a -> DataSet -> Double -> Int -> SVMSolution
      
      createKernelMatrix a x = KernelMatrix matrix
               where matrix = listArray (1, dim) [eval i j | j <- indices x, i <- range(1,j)]
                     dim = ((n+1) * n) `quot` 2
                     eval i j | (i /= j) = evalKernel a (x!i) (x!j)
                              | otherwise = evalKernel a (x!i) (x!j) + dcost a
                     n = snd $ bounds x
      
      simulate a (SVMSolution alpha sv b) points = [(eval p) + b | p <- elems points]
               where eval x = mDot alpha $ listArray (bounds sv) [evalKernel a x v | v <- elems sv]
      
      solve svm (DataSet points values) epsilon maxIter = SVMSolution alpha points b
		where b = (mSum v) / (mSum nu)
		      alpha = mZipWith (\x y -> x - b*y) v nu
		      nu = cga startx ones ones kernel epsilon maxIter
		      v = cga startx values values kernel epsilon maxIter
		      ones = listArray (1, n) $ replicate n 1
		      startx = listArray (1, n) $ replicate n 0
		      n = snd $ bounds values
		      kernel = createKernelMatrix svm points
   
   {- |A least squares support vector machine.  The cost represents the relative expense of missing a
   training versus a more complicated generating function.  The higher this number the better the fit
   of the training set, but at a cost of poorer generalization.  The LSSVM uses every training point
   in the solution and performs least squares regression on the dual of the problem. -}
   data LSSVM = LSSVM {kf::KernelFunction,      -- ^The kernel function defines the feature space.
                       cost::Double,            -- ^The cost coefficient in the Lagrangian.
                       params::[Double]         -- ^Any parameters needed by the 'KernelFunction'.
                      }
 
   instance SVM LSSVM where
      dcost = (0.5 /) . cost
      evalKernel (LSSVM (KernelFunction kf) _ params) = kf params
   
   -- |The conjugate gradient algorithm is used to find the optimal solution.  It will run until a
   -- cutoff delta is reached or for a max number of iterations.
   cga :: DoubleArray -> DoubleArray -> DoubleArray -> KernelMatrix -> Double -> Int -> DoubleArray
   cga x p r k epsilon max_iter = cgahelp x p r norm max_iter False
            where norm = mDot r r
                  cgahelp x _ _ _ _ True = x
                  cgahelp x p r delta iter _ = cgahelp next_x next_p next_r next_delta (iter-1) stop
                           where stop = (next_delta < epsilon * norm) || (iter == 0)
                                 next_x = mAdd x $ scalarmult alpha p
                                 next_p = mAdd next_r $ scalarmult (next_delta/delta) p
                                 next_r = mAdd r $ scalarmult (negate alpha) vector
                                 vector = matmult k p
                                 next_delta = mDot next_r next_r
                                 alpha = delta / (mDot p vector)
   
   -- The following functions are used internally for all of the linear algebra involving kernel
   -- matrices or unboxed arrays of doubles (representing vectors).
   
   -- |Matrix multiplication between a kernel matrix and a vector is handled by this funciton.  Only
   -- the bottom half of the matrix is stored.  This function requires 1 based indices for both of
   -- its arguments.
   matmult :: KernelMatrix -> DoubleArray -> DoubleArray
   matmult (KernelMatrix k) v = listArray (1, d) $ helper 1 1
            where d = snd $ bounds v
                  helper i pos | (i < d) = cpsdot 0 1 pos : helper (i+1) (pos+i)
                               | otherwise = [cpsdot 0 1 pos]
                           where cpsdot acc j n | (j < i) = cpsdot (acc + k!n * v!j) (j+1) (n+1)
                                                | (j < d) = cpsdot (acc + k!n * v!j) (j+1) (n+j)
                                                | otherwise = acc + k!n * v!j
   
   -- |Scalar multiplication of an unboxed array.
   scalarmult :: Double -> DoubleArray -> DoubleArray
   scalarmult = amap . (*)
   
   -- |A version of zipWith for use with unboxed arrays.
   mZipWith :: (Double -> Double -> Double) -> DoubleArray -> DoubleArray -> DoubleArray
   mZipWith f v1 v2 = array (bounds v1) [(i, f (v1!i) (v2!i)) | i <- indices v1]

   -- |Sum the elements of an unboxed array.
   mSum :: DoubleArray -> Double
   mSum = foldl' (+) 0 . elems
   
   -- |Standard dot product of two unboxed arrays.
   mDot :: DoubleArray -> DoubleArray -> Double
   mDot = (mSum .) . mZipWith (*)

   -- |Add two unboxed arrays element by element.
   mAdd :: DoubleArray -> DoubleArray -> DoubleArray
   mAdd = mZipWith (+)
