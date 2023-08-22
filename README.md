# -computer_hardware
Trained a random forest regressor model on computer hardware dataset to predict relative performance values
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "dbec6550",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "63470e79",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"machine.data\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c48468e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 208 entries, 0 to 207\n",
      "Data columns (total 10 columns):\n",
      " #   Column   Non-Null Count  Dtype \n",
      "---  ------   --------------  ----- \n",
      " 0   adviser  208 non-null    object\n",
      " 1   32/60    208 non-null    object\n",
      " 2   125      208 non-null    int64 \n",
      " 3   256      208 non-null    int64 \n",
      " 4   6000     208 non-null    int64 \n",
      " 5   256.1    208 non-null    int64 \n",
      " 6   16       208 non-null    int64 \n",
      " 7   128      208 non-null    int64 \n",
      " 8   198      208 non-null    int64 \n",
      " 9   199      208 non-null    int64 \n",
      "dtypes: int64(8), object(2)\n",
      "memory usage: 16.4+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0ad17a91",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "208"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0d5a5d6d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 208 entries, 0 to 207\n",
      "Data columns (total 10 columns):\n",
      " #   Column   Non-Null Count  Dtype \n",
      "---  ------   --------------  ----- \n",
      " 0   adviser  208 non-null    object\n",
      " 1   32/60    208 non-null    object\n",
      " 2   125      208 non-null    int64 \n",
      " 3   256      208 non-null    int64 \n",
      " 4   6000     208 non-null    int64 \n",
      " 5   256.1    208 non-null    int64 \n",
      " 6   16       208 non-null    int64 \n",
      " 7   128      208 non-null    int64 \n",
      " 8   198      208 non-null    int64 \n",
      " 9   199      208 non-null    int64 \n",
      "dtypes: int64(8), object(2)\n",
      "memory usage: 16.4+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4045f7bd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>125</th>\n",
       "      <th>256</th>\n",
       "      <th>6000</th>\n",
       "      <th>256.1</th>\n",
       "      <th>16</th>\n",
       "      <th>128</th>\n",
       "      <th>198</th>\n",
       "      <th>199</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>208.000000</td>\n",
       "      <td>208.000000</td>\n",
       "      <td>208.000000</td>\n",
       "      <td>208.000000</td>\n",
       "      <td>208.000000</td>\n",
       "      <td>208.000000</td>\n",
       "      <td>208.000000</td>\n",
       "      <td>208.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>204.201923</td>\n",
       "      <td>2880.538462</td>\n",
       "      <td>11824.019231</td>\n",
       "      <td>24.096154</td>\n",
       "      <td>4.644231</td>\n",
       "      <td>17.740385</td>\n",
       "      <td>105.177885</td>\n",
       "      <td>98.850962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>260.833016</td>\n",
       "      <td>3883.839300</td>\n",
       "      <td>11747.916663</td>\n",
       "      <td>37.417999</td>\n",
       "      <td>6.787198</td>\n",
       "      <td>24.913375</td>\n",
       "      <td>161.090223</td>\n",
       "      <td>154.974961</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>17.000000</td>\n",
       "      <td>64.000000</td>\n",
       "      <td>64.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>15.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>50.000000</td>\n",
       "      <td>768.000000</td>\n",
       "      <td>4000.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>28.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>110.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>8000.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>49.500000</td>\n",
       "      <td>45.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>225.000000</td>\n",
       "      <td>4000.000000</td>\n",
       "      <td>16000.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>111.500000</td>\n",
       "      <td>99.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1500.000000</td>\n",
       "      <td>32000.000000</td>\n",
       "      <td>64000.000000</td>\n",
       "      <td>256.000000</td>\n",
       "      <td>52.000000</td>\n",
       "      <td>176.000000</td>\n",
       "      <td>1150.000000</td>\n",
       "      <td>1238.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               125           256          6000       256.1          16  \\\n",
       "count   208.000000    208.000000    208.000000  208.000000  208.000000   \n",
       "mean    204.201923   2880.538462  11824.019231   24.096154    4.644231   \n",
       "std     260.833016   3883.839300  11747.916663   37.417999    6.787198   \n",
       "min      17.000000     64.000000     64.000000    0.000000    0.000000   \n",
       "25%      50.000000    768.000000   4000.000000    0.000000    1.000000   \n",
       "50%     110.000000   2000.000000   8000.000000    8.000000    2.000000   \n",
       "75%     225.000000   4000.000000  16000.000000   32.000000    6.000000   \n",
       "max    1500.000000  32000.000000  64000.000000  256.000000   52.000000   \n",
       "\n",
       "              128          198          199  \n",
       "count  208.000000   208.000000   208.000000  \n",
       "mean    17.740385   105.177885    98.850962  \n",
       "std     24.913375   161.090223   154.974961  \n",
       "min      0.000000     6.000000    15.000000  \n",
       "25%      5.000000    27.000000    28.000000  \n",
       "50%      8.000000    49.500000    45.000000  \n",
       "75%     24.000000   111.500000    99.500000  \n",
       "max    176.000000  1150.000000  1238.000000  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "968ca0c1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='198', ylabel='199'>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(data=data, x=\"198\", y=\"199\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cb2650d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "x = data.drop(columns=[\"adviser\",\"32/60\",\"198\",\"199\"],axis=1)\n",
    "y = data[\"198\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "902de2db",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train,  y_test = train_test_split(x,y,test_size=0.1,random_state=101)\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2f8eb50d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=5, estimator=ElasticNet(),\n",
       "             param_grid={'alpha': [0.1, 5, 10, 100],\n",
       "                         'l1_ratio': [0.1, 0.5, 0.7, 0.9]},\n",
       "             scoring='neg_mean_squared_error', verbose=1)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import ElasticNet\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "model = ElasticNet()\n",
    "param_grid = {'l1_ratio':[0.1, 0.5, 0.7, 0.9], 'alpha':[0.1, 5, 10,100]}\n",
    "grid_search= GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)\n",
    "grid_search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e3bdf252",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 16 candidates, totalling 80 fits\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=5, estimator=ElasticNet(),\n",
       "             param_grid={'alpha': [0.1, 5, 10, 100],\n",
       "                         'l1_ratio': [0.1, 0.5, 0.7, 0.9]},\n",
       "             scoring='neg_mean_squared_error', verbose=1)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#y_train = y_train.values.reshape(1,-1)\n",
    "len(y_train)\n",
    "grid_search.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a7b89c73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ElasticNet(alpha=0.001)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid_search.best_params_\n",
    "\n",
    "model = ElasticNet(alpha=0.001, l1_ratio=0.5)\n",
    "model.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "793daa51",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "19ed8f85",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_absolute_error, mean_squared_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "8b1d2cd5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "34.783880910099434"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_error = mean_absolute_error(y_test,y_pred)\n",
    "mean_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d68447a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "error = mean_squared_error(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d3884e67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "44.82534770985206"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "error**(1/2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "14075464",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "bcaf9325",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor()"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2 = RandomForestRegressor()\n",
    "model2.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "5d219899",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_preds2 = model2.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "d4ed1d1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "error2 = mean_squared_error(y_test, y_preds2)**(1/2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "16c276ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "29.01693894457615"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "error2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "8916a38b",
   "metadata": {},
   "outputs": [],
   "source": [
    "mean_error2 = mean_absolute_error(y_test, y_preds2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "3daadc96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "20.960224489795916"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_error2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "6bc5d2ec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "19.928357150786447"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(mean_error2/y.mean())*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e0daa5f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

