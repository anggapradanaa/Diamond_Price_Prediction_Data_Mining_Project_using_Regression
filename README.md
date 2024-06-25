# 1. BUSINESS UNDERSTANDING

Tujuan: Memprediksi harga diamond

# 2. DATA UNDERSTANDING

1. y (variabel target): price
2. x (variabel lainnya)

### ⚔️Importing Libraries


```python

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn. linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
```

### 🛠️Loading Data


```python
df = pd.read_csv(r"C:\Users\myasu\Downloads\diamonds.csv\diamonds.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>53935</th>
      <td>53936</td>
      <td>0.72</td>
      <td>Ideal</td>
      <td>D</td>
      <td>SI1</td>
      <td>60.8</td>
      <td>57.0</td>
      <td>2757</td>
      <td>5.75</td>
      <td>5.76</td>
      <td>3.50</td>
    </tr>
    <tr>
      <th>53936</th>
      <td>53937</td>
      <td>0.72</td>
      <td>Good</td>
      <td>D</td>
      <td>SI1</td>
      <td>63.1</td>
      <td>55.0</td>
      <td>2757</td>
      <td>5.69</td>
      <td>5.75</td>
      <td>3.61</td>
    </tr>
    <tr>
      <th>53937</th>
      <td>53938</td>
      <td>0.70</td>
      <td>Very Good</td>
      <td>D</td>
      <td>SI1</td>
      <td>62.8</td>
      <td>60.0</td>
      <td>2757</td>
      <td>5.66</td>
      <td>5.68</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>53938</th>
      <td>53939</td>
      <td>0.86</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI2</td>
      <td>61.0</td>
      <td>58.0</td>
      <td>2757</td>
      <td>6.15</td>
      <td>6.12</td>
      <td>3.74</td>
    </tr>
    <tr>
      <th>53939</th>
      <td>53940</td>
      <td>0.75</td>
      <td>Ideal</td>
      <td>D</td>
      <td>SI2</td>
      <td>62.2</td>
      <td>55.0</td>
      <td>2757</td>
      <td>5.83</td>
      <td>5.87</td>
      <td>3.64</td>
    </tr>
  </tbody>
</table>
<p>53940 rows × 11 columns</p>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 53940 entries, 0 to 53939
    Data columns (total 11 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Unnamed: 0  53940 non-null  int64  
     1   carat       53940 non-null  float64
     2   cut         53940 non-null  object 
     3   color       53940 non-null  object 
     4   clarity     53940 non-null  object 
     5   depth       53940 non-null  float64
     6   table       53940 non-null  float64
     7   price       53940 non-null  int64  
     8   x           53940 non-null  float64
     9   y           53940 non-null  float64
     10  z           53940 non-null  float64
    dtypes: float64(6), int64(2), object(3)
    memory usage: 4.5+ MB
    


```python
#menampilkan 10 sample acak
df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7717</th>
      <td>7718</td>
      <td>0.32</td>
      <td>Ideal</td>
      <td>H</td>
      <td>VS1</td>
      <td>61.0</td>
      <td>56.0</td>
      <td>580</td>
      <td>4.43</td>
      <td>4.45</td>
      <td>2.71</td>
    </tr>
    <tr>
      <th>17681</th>
      <td>17682</td>
      <td>1.44</td>
      <td>Very Good</td>
      <td>H</td>
      <td>SI2</td>
      <td>58.2</td>
      <td>61.0</td>
      <td>7128</td>
      <td>7.45</td>
      <td>7.51</td>
      <td>4.35</td>
    </tr>
    <tr>
      <th>51917</th>
      <td>51918</td>
      <td>0.70</td>
      <td>Very Good</td>
      <td>G</td>
      <td>VS2</td>
      <td>59.0</td>
      <td>60.0</td>
      <td>2437</td>
      <td>5.76</td>
      <td>5.83</td>
      <td>3.42</td>
    </tr>
    <tr>
      <th>52493</th>
      <td>52494</td>
      <td>0.60</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VVS1</td>
      <td>62.0</td>
      <td>55.5</td>
      <td>2524</td>
      <td>5.37</td>
      <td>5.42</td>
      <td>3.35</td>
    </tr>
    <tr>
      <th>5092</th>
      <td>5093</td>
      <td>0.90</td>
      <td>Good</td>
      <td>D</td>
      <td>SI1</td>
      <td>63.6</td>
      <td>63.0</td>
      <td>3755</td>
      <td>6.03</td>
      <td>6.07</td>
      <td>3.85</td>
    </tr>
    <tr>
      <th>16542</th>
      <td>16543</td>
      <td>1.22</td>
      <td>Premium</td>
      <td>H</td>
      <td>VS2</td>
      <td>62.3</td>
      <td>58.0</td>
      <td>6608</td>
      <td>6.85</td>
      <td>6.77</td>
      <td>4.24</td>
    </tr>
    <tr>
      <th>26461</th>
      <td>26462</td>
      <td>2.02</td>
      <td>Ideal</td>
      <td>I</td>
      <td>VS2</td>
      <td>61.5</td>
      <td>58.0</td>
      <td>16018</td>
      <td>8.19</td>
      <td>8.11</td>
      <td>5.01</td>
    </tr>
    <tr>
      <th>28657</th>
      <td>28658</td>
      <td>0.31</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VVS1</td>
      <td>60.2</td>
      <td>58.0</td>
      <td>676</td>
      <td>4.37</td>
      <td>4.43</td>
      <td>2.65</td>
    </tr>
    <tr>
      <th>376</th>
      <td>377</td>
      <td>1.20</td>
      <td>Fair</td>
      <td>F</td>
      <td>I1</td>
      <td>64.6</td>
      <td>56.0</td>
      <td>2809</td>
      <td>6.73</td>
      <td>6.66</td>
      <td>4.33</td>
    </tr>
    <tr>
      <th>9580</th>
      <td>9581</td>
      <td>1.08</td>
      <td>Premium</td>
      <td>G</td>
      <td>SI2</td>
      <td>62.9</td>
      <td>59.0</td>
      <td>4627</td>
      <td>6.57</td>
      <td>6.53</td>
      <td>4.12</td>
    </tr>
  </tbody>
</table>
</div>



### Deskripsi Data

Dataset berikut berisi informasi tentang harga berlian dan atribut lainnya.

~ carat (0.2-5.01): Carat adalah berat fisik berlian yang diukur dalam carat metrik. Satu carat sama dengan 0.20 gram dan dibagi menjadi 100 poin.

~ cut (Fair, Good, Very Good, Premium, Ideal): Kualitas potongan. Semakin presisi potongan berlian, semakin memikat berlian tersebut di mata sehingga dinilai dengan nilai tinggi.

~ color (dari J (worst) hingga D (best)): Warna berlian berkualitas permata muncul dalam berbagai nuansa. Dalam rentang dari tidak berwarna hingga kuning muda atau coklat muda. Berlian yang tidak berwarna adalah yang paling langka. Warna alami lainnya (seperti biru, merah, pink) dikenal sebagai "fancy," dan penilaian warnanya berbeda dari berlian putih yang tidak berwarna.

~ clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)): Kejernihan berlian, yang menunjukkan seberapa bebas berlian dari inklusi (ketidaksempurnaan internal) dan cacat (ketidaksempurnaan eksternal).

~ depth (43-79): Ini adalah persentase kedalaman total yang setara dengan z / mean(x, y) = 2 * z / (x + y). Kedalaman berlian adalah tingginya (dalam milimeter) yang diukur dari culet (ujung bawah) hingga meja (permukaan atas datar) seperti yang disebutkan dalam diagram berlabel di atas.

~ table (43-95): Lebar bagian atas berlian (meja) sebagai persentase dari lebar rata-rata. Meja berlian yang ideal berkontribusi pada kecemerlangan berlian.

~ price ($326 - $18826): Ini adalah harga berlian dalam dolar AS. Ini adalah kolom target kita dalam dataset ini.

~ x (0 - 10.74): Panjang berlian (dalam mm).

~ y (0 - 58.9): Lebar berlian (dalam mm).

~ z (0 - 31.8): Kedalaman berlian (dalam mm).

# DATA PREPOCESSING

### Memeriksa nilai dan variabel kategori yang hilang


```python
#Checking missing value in dataset
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 53940 entries, 0 to 53939
    Data columns (total 11 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Unnamed: 0  53940 non-null  int64  
     1   carat       53940 non-null  float64
     2   cut         53940 non-null  object 
     3   color       53940 non-null  object 
     4   clarity     53940 non-null  object 
     5   depth       53940 non-null  float64
     6   table       53940 non-null  float64
     7   price       53940 non-null  int64  
     8   x           53940 non-null  float64
     9   y           53940 non-null  float64
     10  z           53940 non-null  float64
    dtypes: float64(6), int64(2), object(3)
    memory usage: 4.5+ MB
    

Catatan :
Terdapat total data adalah 53940, berdasarkan informasi mengenai jumlah isian perkolom, terlihat bahwa jumlah baris adalah 53940. Jadi data tersebut tidak memiliki missing value. 

Tipe data (cut, color, dan clarity) adalah object, sehingga perlu di convert menjadi variabel numerik Sebelum kita memasukkan data ke dalam algoritma. 

### Mengevaluasi fitur kategorikal

Melihat distribusi kolom cut,color, dan clarity terhadap price


```python
plt.figure(figsize=(10,9))
cols = sns.color_palette("coolwarm", 5)
ax = sns.violinplot(x="cut", y="price", data=df, palette=cols, scale="count")
ax.set_title("Diamond Cut for Price", color="#774571", fontsize=20)
ax.set_ylabel("Price", color="#4e4c39", fontsize=15)
ax.set_xlabel("Cut", color="#4e4c39", fontsize=15)

# Menambahkan label pada masing-masing kotak di plot violin
for i in range(len(ax.collections) // 2):
    med = df.groupby("cut")["price"].median().values[i]
    x = i
    y = med
    ax.text(x, y, f'{y:.2f}', ha='center', va='center', fontsize=12, color='black')

plt.show()

```


    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Diamond%20Cut%20for%20Price.png'>
    



```python
plt.figure(figsize=(12,9))
# Menggunakan palet warna 'coolwarm' dari seaborn
cols = sns.color_palette("coolwarm", 7)
ax = sns.violinplot(x="color", y="price", data=df, palette=cols, scale="count")
ax.set_title("Diamond Colors for Price", color="#774571", fontsize=20)
ax.set_ylabel("Price", color="#4e4c39", fontsize=15)
ax.set_xlabel("Color", color="#4e4c39", fontsize=15)

# Menambahkan label pada masing-masing kotak di plot violin
for i in range(len(ax.collections) // 2):
    med = df.groupby("color")["price"].median().values[i]
    x = i
    y = med
    ax.text(x, y, f'{y:.2f}', ha='center', va='center', fontsize=12, color='black')

plt.show()

```


    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Diamond%20Colors%20for%20Price.png'>
    



```python
plt.figure(figsize=(13,8))
# Menggunakan palet warna 'viridis' dari seaborn
cols = sns.color_palette("viridis", 8)
ax = sns.violinplot(x="clarity", y="price", data=df, palette=cols, scale="count")
ax.set_title("Diamond Clarity for Price", color="#774571", fontsize=20)
ax.set_ylabel("Price", color="#4e4c39", fontsize=15)
ax.set_xlabel("Clarity", color="#4e4c39", fontsize=15)

# Menambahkan label pada masing-masing kotak di plot violin
for i in range(len(ax.collections) // 2):
    med = df.groupby("clarity")["price"].median().values[i]
    x = i
    y = med
    ax.text(x, y, f'{y:.2f}', ha='center', va='center', fontsize=12, color='black')

plt.show()

```


    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Diamond%20Clarity%20for%20Price.png'>
    


Catatan :
Potongan berlian "Ideal" adalah yang paling banyak jumlahnya, sedangkan potongan "Fair" adalah yang paling sedikit. 

Berlian dengan warna "J", yang merupakan yang terburuk, adalah yang paling langka, namun berlian dengan warna "H" dan "G" lebih banyak jumlahnya meskipun kualitasnya juga lebih rendah.

Berlian dengan kejernihan "IF" yang merupakan yang terbaik serta "I1" yang merupakan yang terburuk sangat langka, sedangkan sisanya sebagian besar memiliki kejernihan yang berada di antara keduanya.

### Statistik Deskriptif


```python
# Melakukan Analisis Univariat untuk deskripsi statistik dan pemahaman tentang sebaran data
descriptive_stats = df.describe().T
descriptive_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>53940.0</td>
      <td>26970.500000</td>
      <td>15571.281097</td>
      <td>1.0</td>
      <td>13485.75</td>
      <td>26970.50</td>
      <td>40455.25</td>
      <td>53940.00</td>
    </tr>
    <tr>
      <th>carat</th>
      <td>53940.0</td>
      <td>0.797940</td>
      <td>0.474011</td>
      <td>0.2</td>
      <td>0.40</td>
      <td>0.70</td>
      <td>1.04</td>
      <td>5.01</td>
    </tr>
    <tr>
      <th>depth</th>
      <td>53940.0</td>
      <td>61.749405</td>
      <td>1.432621</td>
      <td>43.0</td>
      <td>61.00</td>
      <td>61.80</td>
      <td>62.50</td>
      <td>79.00</td>
    </tr>
    <tr>
      <th>table</th>
      <td>53940.0</td>
      <td>57.457184</td>
      <td>2.234491</td>
      <td>43.0</td>
      <td>56.00</td>
      <td>57.00</td>
      <td>59.00</td>
      <td>95.00</td>
    </tr>
    <tr>
      <th>price</th>
      <td>53940.0</td>
      <td>3932.799722</td>
      <td>3989.439738</td>
      <td>326.0</td>
      <td>950.00</td>
      <td>2401.00</td>
      <td>5324.25</td>
      <td>18823.00</td>
    </tr>
    <tr>
      <th>x</th>
      <td>53940.0</td>
      <td>5.731157</td>
      <td>1.121761</td>
      <td>0.0</td>
      <td>4.71</td>
      <td>5.70</td>
      <td>6.54</td>
      <td>10.74</td>
    </tr>
    <tr>
      <th>y</th>
      <td>53940.0</td>
      <td>5.734526</td>
      <td>1.142135</td>
      <td>0.0</td>
      <td>4.72</td>
      <td>5.71</td>
      <td>6.54</td>
      <td>58.90</td>
    </tr>
    <tr>
      <th>z</th>
      <td>53940.0</td>
      <td>3.538734</td>
      <td>0.705699</td>
      <td>0.0</td>
      <td>2.91</td>
      <td>3.53</td>
      <td>4.04</td>
      <td>31.80</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Melihat mean dan median dari kolom 'price'
mean_price = descriptive_stats.loc['price', 'mean']
median_price = df['price'].median()

print(f"Mean price: {mean_price}")
print(f"Median price: {median_price}")

if mean_price > median_price:
    print("Distribusi harga condong ke kanan.")
else:
    print("Distribusi harga tidak condong ke kanan.")

# Menghitung skewness
price_skewness = df['price'].skew()
print(f"Skewness dari harga: {price_skewness}")

if price_skewness > 0:
    print("Distribusi harga condong ke kanan.")
else:
    print("Distribusi harga tidak condong ke kanan.")

```

    Mean price: 3932.799721913237
    Median price: 2401.0
    Distribusi harga condong ke kanan.
    Skewness dari harga: 1.618395283383529
    Distribusi harga condong ke kanan.
    


```python
# Melihat nilai minimum dari kolom 'x', 'y', dan 'z'
min_x = descriptive_stats.loc['x', 'min']
min_y = descriptive_stats.loc['y', 'min']
min_z = descriptive_stats.loc['z', 'min']

print(f"Nilai minimum untuk 'x': {min_x}")
print(f"Nilai minimum untuk 'y': {min_y}")
print(f"Nilai minimum untuk 'z': {min_z}")

# Menggunakan .min() untuk konfirmasi
print(df[['x', 'y', 'z']].min())

# Mengidentifikasi baris dengan nilai 0 pada kolom 'x', 'y', atau 'z'
rows_with_zero = df[(df['x'] == 0) | (df['y'] == 0) | (df['z'] == 0)]
print(rows_with_zero)

# Menyimpulkan apakah ada nilai yang tidak masuk akal
if (min_x == 0) or (min_y == 0) or (min_z == 0):
    print("Ada nilai 0 pada kolom 'x', 'y', atau 'z' yang membuat data tidak masuk akal.")
else:
    print("Tidak ada nilai 0 pada kolom 'x', 'y', atau 'z'.")
```

    Nilai minimum untuk 'x': 0.0
    Nilai minimum untuk 'y': 0.0
    Nilai minimum untuk 'z': 0.0
    x    0.0
    y    0.0
    z    0.0
    dtype: float64
           Unnamed: 0  carat        cut color clarity  depth  table  price     x  \
    2207         2208   1.00    Premium     G     SI2   59.1   59.0   3142  6.55   
    2314         2315   1.01    Premium     H      I1   58.1   59.0   3167  6.66   
    4791         4792   1.10    Premium     G     SI2   63.0   59.0   3696  6.50   
    5471         5472   1.01    Premium     F     SI2   59.2   58.0   3837  6.50   
    10167       10168   1.50       Good     G      I1   64.0   61.0   4731  7.15   
    11182       11183   1.07      Ideal     F     SI2   61.6   56.0   4954  0.00   
    11963       11964   1.00  Very Good     H     VS2   63.3   53.0   5139  0.00   
    13601       13602   1.15      Ideal     G     VS2   59.2   56.0   5564  6.88   
    15951       15952   1.14       Fair     G     VS1   57.5   67.0   6381  0.00   
    24394       24395   2.18    Premium     H     SI2   59.4   61.0  12631  8.49   
    24520       24521   1.56      Ideal     G     VS2   62.2   54.0  12800  0.00   
    26123       26124   2.25    Premium     I     SI1   61.3   58.0  15397  8.52   
    26243       26244   1.20    Premium     D    VVS1   62.1   59.0  15686  0.00   
    27112       27113   2.20    Premium     H     SI1   61.2   59.0  17265  8.42   
    27429       27430   2.25    Premium     H     SI2   62.8   59.0  18034  0.00   
    27503       27504   2.02    Premium     H     VS2   62.7   53.0  18207  8.02   
    27739       27740   2.80       Good     G     SI2   63.8   58.0  18788  8.90   
    49556       49557   0.71       Good     F     SI2   64.1   60.0   2130  0.00   
    49557       49558   0.71       Good     F     SI2   64.1   60.0   2130  0.00   
    51506       51507   1.12    Premium     G      I1   60.4   59.0   2383  6.71   
    
              y    z  
    2207   6.48  0.0  
    2314   6.60  0.0  
    4791   6.47  0.0  
    5471   6.47  0.0  
    10167  7.04  0.0  
    11182  6.62  0.0  
    11963  0.00  0.0  
    13601  6.83  0.0  
    15951  0.00  0.0  
    24394  8.45  0.0  
    24520  0.00  0.0  
    26123  8.42  0.0  
    26243  0.00  0.0  
    27112  8.37  0.0  
    27429  0.00  0.0  
    27503  7.95  0.0  
    27739  8.85  0.0  
    49556  0.00  0.0  
    49557  0.00  0.0  
    51506  6.67  0.0  
    Ada nilai 0 pada kolom 'x', 'y', atau 'z' yang membuat data tidak masuk akal.
    

Catatan :
1. "Price" sesuai perkiraan condong ke kanan, dengan lebih banyak titik data di sebelah kiri.
2. Pada fitur dimensional 'x', 'y', & 'z', nilai minimum adalah 0 sehingga membuat titik data tersebut menjadi objek berlian 1D atau 2D yang tidak masuk akal - sehingga perlu diimputasi dengan nilai yang sesuai atau dihapus seluruhnya.


```python
# Melakukan Analisis Bivariat dengan memeriksa pairplot
ax = sns.pairplot(df, hue="cut", palette=cols)

```


    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Analisis%20Bivariat.png'>
    


Catatan:
1. Terdapat fitur "unnamed" yang tidak berguna, yang merupakan indeks dan perlu dihilangkan.
2. Terdapat outlier yang perlu ditangani karena dapat mempengaruhi kinerja model
3. Kolom "y" dan "z" memiliki beberapa outlier dimensional dalam dataset dan perlu dihilangkan.
4. Kolom "depth" & "table" seharusnya dibatasi setelah diperiksa Plot Garis.

### Memeriksa Potensi Outlier


```python
lm = sns.lmplot(x = 'price', y= 'y', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs y', color = '#774571', fontsize = 15)
plt.show()
```


    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Price%20vs%20y.png'>
    



```python
lm = sns.lmplot(x = 'price', y= 'z', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs z', color = '#774571', fontsize = 15)
plt.show()
```


    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Price%20vs%20z.png'>
    



```python
lm = sns.lmplot(x = 'price', y= 'depth', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs depth', color = '#774571', fontsize = 15)
plt.show()
```


    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Price%20vs%20Depth.png'>
    



```python
lm = sns.lmplot(x = 'price', y= 'table', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs table', color = '#774571', fontsize = 15)
plt.show()
```


    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Price%20vs%20table.png'>
    


Catatan:
Dengan melakukan plot di atas, outlier dapat dilihat dengan mudah.

### Pembersihan Data


```python
#Menghapus fitur "Unnamed"
df = df.drop(["Unnamed: 0"], axis=1)
df.shape
```




    (53940, 10)




```python
#Menghapus titik data yang memiliki nilai minimum 0 pada salah satu fitur x, y, atau z. 
df = df.drop(df[df["x"]==0].index)
df = df.drop(df[df["y"]==0].index)
df = df.drop(df[df["z"]==0].index)
df.shape
```




    (53920, 10)



### Menghapus Ouliers


```python
#Menghapus outlier (karena memiliki dataset yang besar) dengan menentukan langkah-langkah yang sesuai di seluruh fiturdf = data_df[(data_df["depth"]<75)&(data_df["depth"]>45)]
df = df[(df["depth"]<75)&(df["depth"]>45)]
df = df[(df["table"]<80)&(df["table"]>40)]
df = df[(df["x"]<40)]
df = df[(df["y"]<40)]
df = df[(df["z"]<40)&(df["z"]>2)]
df.shape
```




    (53909, 10)



### Encoding Variabel Kategorik


```python
#Membuat salinan untuk menjaga data asli dalam bentuknya yang utuh
df1 = df.copy()

#Menerapkan label encoder pada kolom-kolom dengan data kategorikal
columns = ['cut','color','clarity']
label_encoder = LabelEncoder()
for col in columns:
    df1[col] = label_encoder.fit_transform(df1[col])
df1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>53909.000000</td>
      <td>53909.000000</td>
      <td>53909.000000</td>
      <td>53909.000000</td>
      <td>53909.000000</td>
      <td>53909.000000</td>
      <td>53909.000000</td>
      <td>53909.000000</td>
      <td>53909.000000</td>
      <td>53909.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.797617</td>
      <td>2.553396</td>
      <td>2.593964</td>
      <td>3.835575</td>
      <td>61.749743</td>
      <td>57.455852</td>
      <td>3930.513680</td>
      <td>5.731441</td>
      <td>5.733764</td>
      <td>3.539994</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.473759</td>
      <td>1.027444</td>
      <td>1.701283</td>
      <td>1.724540</td>
      <td>1.420093</td>
      <td>2.226169</td>
      <td>3987.145802</td>
      <td>1.119369</td>
      <td>1.116891</td>
      <td>0.702085</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>50.800000</td>
      <td>43.000000</td>
      <td>326.000000</td>
      <td>3.730000</td>
      <td>3.680000</td>
      <td>2.060000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.400000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>61.000000</td>
      <td>56.000000</td>
      <td>949.000000</td>
      <td>4.710000</td>
      <td>4.720000</td>
      <td>2.910000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.700000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>61.800000</td>
      <td>57.000000</td>
      <td>2400.000000</td>
      <td>5.700000</td>
      <td>5.710000</td>
      <td>3.530000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.040000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>62.500000</td>
      <td>59.000000</td>
      <td>5322.000000</td>
      <td>6.540000</td>
      <td>6.540000</td>
      <td>4.040000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.010000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>7.000000</td>
      <td>73.600000</td>
      <td>79.000000</td>
      <td>18823.000000</td>
      <td>10.740000</td>
      <td>31.800000</td>
      <td>31.800000</td>
    </tr>
  </tbody>
</table>
</div>



Catatan:

Setelah fitur-fitur kategorikal dikonversi menjadi kolom-kolom numerik, kita juga mendapatkan ringkasan 5 poin bersama dengan jumlah, rata-rata, dan standar deviasi untuk mereka. 

Sekarang, kita dapat menganalisis matriks korelasi setelah selesai dengan pre-processing untuk pemilihan fitur yang mungkin guna membuat dataset lebih bersih dan optimal sebelum kita masukkan ke dalam algoritma.

### Matriks Kolerasi


```python
#Mengeksaminasi matriks korelasi menggunakan heatmap
cmap = sns.diverging_palette(205, 133, 63, as_cmap=True)
cols = (["#FFCDEA", "#E59BE9", "#FB9AD1", "#BC7FCD", "#D862BC", "#86469C"])
corrmat= df1.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,cmap=cols,annot=True)
```




    <Axes: >




    
<img src = 'https://github.com/anggapradanaa/Sales_Prediction_Data_Mining_Project_using_Regression/blob/main/Matriks%20Korelasi.png'>
    


Catatan:

Fitur "carat", "x", "y", "z" memiliki korelasi yang tinggi dengan variabel target kita, yaitu harga.

Fitur "cut", "clarity", "depth" memiliki korelasi yang sangat rendah (<|0.1|) sehingga mungkin dapat dihapus, meskipun karena hanya ada beberapa fitur yang dipilih, kita tidak akan melakukannya (tetap menggunakannya).

# 4. MODEL BUILDING


```python
# Mendefinisikan variabel independen dan dependen
x = df1.drop(["price"],axis =1)
y = df1["price"]
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.20, random_state=25)
```


```python
# Membangun Pipeline Standar Scaler dan Model untuk Berbagai Regressor

pipeline_lr=Pipeline([("scalar1",StandardScaler()),
                     ("lr",LinearRegression())])

pipeline_lasso=Pipeline([("scalar2", StandardScaler()),
                      ("lasso",Lasso())])

pipeline_dt=Pipeline([("scalar3",StandardScaler()),
                     ("dt",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar4",StandardScaler()),
                     ("rf",RandomForestRegressor())])


pipeline_kn=Pipeline([("scalar5",StandardScaler()),
                     ("kn",KNeighborsRegressor())])


pipeline_xgb=Pipeline([("scalar6",StandardScaler()),
                     ("xgb",XGBRegressor())])

# Daftar semua saluran pipa
pipelines = [pipeline_lr, pipeline_lasso, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipeline_dict = {0: "LinearRegression", 1: "Lasso", 2: "DecisionTree", 3: "RandomForest",4: "KNeighbors", 5: "XGBRegressor"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(x_train, y_train)
```

# 5. EVALUATION


```python
#hasil tiap-tiap jenis model
cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, x_train,y_train,scoring="neg_root_mean_squared_error", cv=12)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipeline_dict[i], -1 * cv_score.mean()))
```

    LinearRegression: 1383.854012 
    Lasso: 1366.991298 
    DecisionTree: 738.917667 
    RandomForest: 548.624640 
    KNeighbors: 816.559263 
    XGBRegressor: 548.346850 
    

## XGBClassifier


```python
# Prediksi model pada data pengujian dengan XGBClassifier yang memberikan kita RMSE paling sedikit
pred = pipeline_xgb.predict(x_test)
print("R^2:",metrics.r2_score(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
```

    R^2: 0.9821291192884176
    Adjusted R^2: 0.9821141881775372
    


```python
# Membuat DataFrame comparison_df dengan data aktual dan prediksi
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})

# Menampilkan tabel comparison_df
print("Tabel Perbandingan Harga Asli dan Harga Prediksi:")
print(comparison_df)

# Menambahkan kolom selisih antara harga asli dan harga prediksi
comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Predicted']

# Menampilkan tabel dengan kolom selisih
print("\nTabel Perbandingan Harga Asli, Harga Prediksi, dan Selisih:")
print(comparison_df)

# Opsional: Menyimpan tabel ke dalam file CSV
comparison_df.to_csv(r'C:\Users\myasu\Downloads\perbandingan harga diamonds.csv', index=False)
```

    Tabel Perbandingan Harga Asli, Harga Prediksi, dan Selisih:
           Actual     Predicted   Difference
    31712     771    860.958082   -89.958082
    19865    8419   8730.078945  -311.078945
    42610     505    521.556190   -16.556190
    29785     709    707.374216     1.625784
    20340    8739   9841.110078 -1102.110078
    ...       ...           ...          ...
    50799    2306   2367.419414   -61.419414
    40238    1124   1187.310403   -63.310403
    23860   11951  11184.547598   766.452402
    11809    5090   4660.840781   429.159219
    39776    1094   1007.533412    86.466588
    
    [10782 rows x 3 columns]
    

## Voting Regressor


```python
# Membuat regressor voting dengan XGBRegressor dan RandomForestRegressor
voting_regressor = VotingRegressor(estimators=[
    ('xgb', XGBRegressor()),
    ('rf', RandomForestRegressor())
])
```


```python
# Standarisasi data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Melatih regressor voting
voting_regressor.fit(x_train_scaled, y_train)

# Memprediksi data tes
pred_voting_regressor = voting_regressor.predict(x_test_scaled)

# Evaluasi model
print("R^2:", metrics.r2_score(y_test, pred_voting_regressor))
print("Adjusted R^2:", 1 - (1 - metrics.r2_score(y_test, pred_voting_regressor)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))
```

    R^2: 0.9832147486730024
    Adjusted R^2: 0.9832007246048681
    


```python
# Membuat DataFrame comparison_df dengan data aktual dan prediksi
comparison_df_voting_regressor = pd.DataFrame({'Actual': y_test, 'Predicted': pred_voting_regressor})

# Menambahkan kolom selisih antara harga asli dan harga prediksi
comparison_df_voting_regressor['Difference'] = comparison_df_voting_regressor['Actual'] - comparison_df_voting_regressor['Predicted']

# Menampilkan tabel comparison_df dengan kolom selisih
print("Tabel Perbandingan Harga Asli, Harga Prediksi, dan Selisih:")
print(comparison_df_voting_regressor)
```

    Tabel Perbandingan Harga Asli, Harga Prediksi, dan Selisih:
           Actual     Predicted   Difference
    31712     771    862.378082   -91.378082
    19865    8419   8712.408945  -293.408945
    42610     505    519.820357   -14.820357
    29785     709    706.599216     2.400784
    20340    8739   9958.460078 -1219.460078
    ...       ...           ...          ...
    50799    2306   2332.244414   -26.244414
    40238    1124   1191.995403   -67.995403
    23860   11951  11354.317598   596.682402
    11809    5090   4632.915781   457.084219
    39776    1094   1008.563412    85.436588
    
    [10782 rows x 3 columns]
    

## Rata-Rata


```python
from sklearn.metrics import mean_squared_error
```


```python
# Membuat model
xgb_model = XGBRegressor()
rf_model = RandomForestRegressor()

# Standarisasi data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Melatih model
xgb_model.fit(x_train_scaled, y_train)
rf_model.fit(x_train_scaled, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div>




```python
# Memprediksi data tes
xgb_pred = xgb_model.predict(x_test_scaled)
rf_pred = rf_model.predict(x_test_scaled)

# Menghitung RMSE masing-masing model
xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

# Menghitung rata-rata prediksi dari kedua model
combined_pred = (xgb_pred + rf_pred) / 2

# Evaluasi model gabungan
print("XGB RMSE:", xgb_rmse)
print("RF RMSE:", rf_rmse)
print("Combined RMSE:", mean_squared_error(y_test, combined_pred, squared=False))
print("R^2:", metrics.r2_score(y_test, combined_pred))
print("Adjusted R^2:", 1 - (1 - metrics.r2_score(y_test, combined_pred)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1))
```

    XGB RMSE: 536.7066529757042
    RF RMSE: 543.5824976698756
    Combined RMSE: 519.8018038674221
    R^2: 0.983237161421462
    Adjusted R^2: 0.9832231560791665
    


```python
# Membuat DataFrame comparison_df dengan data aktual dan prediksi gabungan
comparison_df_combine = pd.DataFrame({'Actual': y_test, 'Predicted': combined_pred})

# Menambahkan kolom selisih antara harga asli dan harga prediksi
comparison_df_combine['Difference'] = comparison_df_combine['Actual'] - comparison_df_combine['Predicted']

# Menampilkan tabel comparison_df dengan kolom selisih
print("\nTabel Perbandingan Harga Asli, Harga Prediksi, dan Selisih:")
print(comparison_df_combine)
```

    
    Tabel Perbandingan Harga Asli, Harga Prediksi, dan Selisih:
           Actual     Predicted   Difference
    31712     771    864.883082   -93.883082
    19865    8419   8731.213945  -312.213945
    42610     505    519.401190   -14.401190
    29785     709    706.899216     2.100784
    20340    8739  10009.770078 -1270.770078
    ...       ...           ...          ...
    50799    2306   2368.699414   -62.699414
    40238    1124   1188.300403   -64.300403
    23860   11951  11251.547598   699.452402
    11809    5090   4659.280781   430.719219
    39776    1094   1004.418412    89.581588
    
    [10782 rows x 3 columns]
    


```python

```


```python

```


```python

```


```python

```
