# Predicting Ames House Prices

Team: Kevin Singh, Ronit Patil, Simmon Chauahan

This project predicts SalePrice for houses in Ames, Iowa.

## The data

The train file has 1,200 houses, 79 features, and SalePrice.

The test file has 260 houses with the same features and no SalePrice.

## The process

### 1. Looking at the data

A scatterplot of GrLivArea against SalePrice showed two core problems. Prices had a long right tail, and they spread out more as houses got bigger. Plotting both on a log scale turned that fan shape into a fairly straight, even band that eased training.

### 2. Outliers

The dataset's author recommends looking at houses over 4,000 sq ft. Three houses in train qualify.

Id 524 is 4,676 sq ft and sold for $184,750 as a partial sale. I removed it because it is far below the trend.

Id 692 is 4,316 sq ft and sold for $755,000 as a normal sale. I kept it because it follows the trend.

Id 1183 is 4,476 sq ft and sold for $745,000 as an abnormal sale. I kept it because it follows the trend.

### 3. Missing values

Most blanks aren't really missing. The data description says a blank in a column like PoolQC means the house has no pool, and I confirmed this against related columns such as PoolArea. Dropping those rows or columns would have thrown away real information.

For 15 text columns, such as PoolQC and GarageType, a blank means the house doesn't have the feature, so I filled them with 'None'.

For MasVnrArea, a blank means no masonry veneer, so I filled it with 0.

For LotFrontage, the value is truly missing, so I used the median of the house's neighborhood.

For GarageYrBlt, a blank means no garage, so I used YearBuilt as a harmless placeholder.

For Electrical, one value is truly missing in the test file only, so I used the most common value.

### 4. Encoding and scaling

Ranked categories like ExterQual were mapped to numbers in order, so Ex scores above Gd.

Unranked categories like Neighborhood were one-hot encoded. The setting handle_unknown='ignore' is needed because test has categories that train doesn't.

MSSubClass looks numeric but is a label, so it was converted to text.

Numeric columns were scaled, because Lasso Estimator treats all coefficients equally.

This grew the data from 79 columns to 245.

### 5. Log transforms

For the target, the model predicts log(SalePrice). Later down the road predictions are converted back with np.exp.

For the features, I logged columns that met a rule set from the shape of the data: more than 50 distinct values, skew above 0.5, and under 5% zeros. That selected LotArea, LotFrontage, GrLivArea and 1stFlrSF.

Columns that are mostly zeros are left alone. Logging them leaves a pile at 0 and a gap before everything else, so it doesn't fix the skew.

### 6. Pipeline

The imputing, encoding, scaling and feature logs all run in one scikit-learn pipeline. That applies the same steps to test automatically, and refits them on training data only, so validation houses don't leak in. The LotFrontage and GarageYrBlt fills sit outside the pipeline because SimpleImputer can't do them.

### 7. Choosing a model

I compared models with 5-fold cross-validation mentioned in the SKLearn docs as a way to evaluate models, averaged over three different shuffles of the folds. These are the log RMSE scores, and lower is better. I chose models based on the SKLearn Cheat Sheet in the github that guided me to the right decision

Lasso + gradient boosting (50/50): 0.1095

Lasso: 0.1116

ElasticNet: 0.1117

Ridge: 0.1132

Gradient boosting: 0.1153

Random forest: 0.1343

### 8. Checking the final model

I held out 20% of train as a validation set. Running this gave a train error of 0.0707 and a validation error of 0.1235.
