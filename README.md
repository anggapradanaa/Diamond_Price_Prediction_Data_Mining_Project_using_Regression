# 1. BUSINESS UNDERSTANDING


```python

```

# 2. DATA UNDERSTANDING

### ⚔️Importing Libraries


```python
!pip install xgboost

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

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

    Requirement already satisfied: xgboost in c:\users\aditya p j\anaconda3\lib\site-packages (2.0.3)
    Requirement already satisfied: numpy in c:\users\aditya p j\anaconda3\lib\site-packages (from xgboost) (1.24.3)
    Requirement already satisfied: scipy in c:\users\aditya p j\anaconda3\lib\site-packages (from xgboost) (1.11.1)
    

### 🛠️Loading Data


```python
df = pd.read_csv(r"C:\Users\Aditya P J\Documents\Python Scripts\Data\diamonds.csv")
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
      <th>52128</th>
      <td>52129</td>
      <td>0.60</td>
      <td>Ideal</td>
      <td>G</td>
      <td>VVS2</td>
      <td>61.3</td>
      <td>56.0</td>
      <td>2467</td>
      <td>5.43</td>
      <td>5.46</td>
      <td>3.34</td>
    </tr>
    <tr>
      <th>20136</th>
      <td>20137</td>
      <td>1.22</td>
      <td>Ideal</td>
      <td>H</td>
      <td>VS2</td>
      <td>61.3</td>
      <td>56.0</td>
      <td>8596</td>
      <td>6.85</td>
      <td>6.88</td>
      <td>4.21</td>
    </tr>
    <tr>
      <th>50552</th>
      <td>50553</td>
      <td>0.73</td>
      <td>Ideal</td>
      <td>F</td>
      <td>SI2</td>
      <td>62.1</td>
      <td>56.0</td>
      <td>2276</td>
      <td>5.77</td>
      <td>5.80</td>
      <td>3.59</td>
    </tr>
    <tr>
      <th>45156</th>
      <td>45157</td>
      <td>0.50</td>
      <td>Ideal</td>
      <td>D</td>
      <td>SI1</td>
      <td>61.6</td>
      <td>55.0</td>
      <td>1654</td>
      <td>5.13</td>
      <td>5.10</td>
      <td>3.15</td>
    </tr>
    <tr>
      <th>12120</th>
      <td>12121</td>
      <td>0.91</td>
      <td>Premium</td>
      <td>D</td>
      <td>VS2</td>
      <td>62.0</td>
      <td>58.0</td>
      <td>5167</td>
      <td>6.20</td>
      <td>6.15</td>
      <td>3.83</td>
    </tr>
    <tr>
      <th>25150</th>
      <td>25151</td>
      <td>1.50</td>
      <td>Premium</td>
      <td>G</td>
      <td>VS1</td>
      <td>60.8</td>
      <td>61.0</td>
      <td>13720</td>
      <td>7.36</td>
      <td>7.29</td>
      <td>4.45</td>
    </tr>
    <tr>
      <th>36556</th>
      <td>36557</td>
      <td>0.40</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>62.2</td>
      <td>58.0</td>
      <td>945</td>
      <td>4.74</td>
      <td>4.71</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>12336</th>
      <td>12337</td>
      <td>1.10</td>
      <td>Ideal</td>
      <td>G</td>
      <td>SI1</td>
      <td>62.3</td>
      <td>56.0</td>
      <td>5226</td>
      <td>6.64</td>
      <td>6.58</td>
      <td>4.12</td>
    </tr>
    <tr>
      <th>3560</th>
      <td>3561</td>
      <td>1.01</td>
      <td>Premium</td>
      <td>H</td>
      <td>SI1</td>
      <td>59.2</td>
      <td>58.0</td>
      <td>3417</td>
      <td>6.58</td>
      <td>6.56</td>
      <td>3.89</td>
    </tr>
    <tr>
      <th>47408</th>
      <td>47409</td>
      <td>0.50</td>
      <td>Ideal</td>
      <td>F</td>
      <td>VVS2</td>
      <td>62.3</td>
      <td>57.0</td>
      <td>1850</td>
      <td>5.06</td>
      <td>5.08</td>
      <td>3.16</td>
    </tr>
  </tbody>
</table>
</div>



### Deskripsi Data

Dataset berikut berisi informasi harga dan atribut lainnya.

~ carat (0.2-5.01): Carat adalah berat fisik berlian yang diukur dalam carat metrik. Satu carat sama dengan 0.20 gram dan dibagi menjadi 100 poin.

~ cut (Fair, Good, Very Good, Premium, Ideal): Kualitas potongan. Semakin presisi potongan berlian, semakin memikat berlian tersebut di mata sehingga dinilai dengan nilai tinggi.

~ color (dari J (worst) hingga D (best)): Warna berlian berkualitas permata muncul dalam berbagai nuansa. Dalam rentang dari tidak berwarna hingga kuning muda atau coklat muda. Berlian yang tidak berwarna adalah yang paling langka. Warna alami lainnya (seperti biru, merah, pink) dikenal sebagai "fancy," dan penilaian warnanya berbeda dari berlian putih yang tidak berwarna.

~ clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)): Berlian dapat memiliki karakteristik internal yang dikenal sebagai inklusi atau karakteristik eksternal yang dikenal sebagai cacat. Berlian tanpa inklusi atau cacat sangat langka; namun, sebagian besar karakteristik hanya dapat dilihat dengan pembesaran.

~ depth (43-79): Ini adalah persentase kedalaman total yang setara dengan z / mean(x, y) = 2 * z / (x + y). Kedalaman berlian adalah tingginya (dalam milimeter) yang diukur dari culet (ujung bawah) hingga meja (permukaan atas datar) seperti yang disebutkan dalam diagram berlabel di atas.

~ table (43-95): Ini adalah lebar bagian atas berlian relatif terhadap titik terlebar. Ini memberikan berlian kilauan dan kecemerlangan yang menakjubkan dengan memantulkan cahaya ke segala arah yang ketika dilihat oleh pengamat, tampak berkilau.

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
Terdapat total data adalah 53940, berdasarkan informasi mengenai jumlah isian perkolom, terlihat bahwa jumlah baris adalah 53840. Jadi data tersebut tidak memiliki missing value. 

Tipe data (cut, color, dan clarity) adalah object, sehingga perlu di convert menjadi variabel numerik Sebelum kita memasukkan data ke dalam algoritma. 

### Mengevaluasi fitur kategorikal


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


    
![png](output_16_0.png)
    



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


    
![png](output_17_0.png)
    



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


    
![png](output_18_0.png)
    


Catatan :
Potongan "Ideal" diamonds adalah yang paling banyak jumlahnya, sedangkan "Fair" diamonds adalah yang paling sedikit jumlahnya. Lebih banyak diamonds dari semua jenis potongan untuk kategori harga yang lebih rendah.

Dengan warna "J" diamonds, yang merupakan yang terburuk, sangat langka, namun "H" dan "G" diamonds lebih banyak jumlahnya meskipun kualitasnya juga rendah.

Dengan kejelasan "IF" diamonds, yang merupakan yang terbaik, serta "I1" diamonds, yang merupakan yang terburuk, sangat langka, sementara yang lainnya sebagian besar memiliki kejelasan di antara keduanya.

### Statistik Deskriptif


```python
# Melakukan Analisis Univariat untuk deskripsi statistik dan pemahaman tentang sebaran data
df.describe().T

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



Catatan :
    
    "Price" seperti yang diharapkan cenderung condong ke kanan, dengan jumlah titik data yang lebih banyak di sebelah kiri.
Di bawah fitur dimensional 'x', 'y', & 'z' - nilai minimum adalah 0 sehingga membuat titik data tersebut menjadi objek berlian 1D atau 2D yang tidak masuk akal - oleh karena itu perlu untuk diimputasi dengan nilai yang sesuai atau dihapus sama sekali.


```python
# Melakukan Analisis Bivariat dengan memeriksa pairplot
ax = sns.pairplot(df, hue="cut", palette=cols)

```


    
![png](output_23_0.png)
    


Catatan:
1. Terdapat fitur "unnamed" yang tidak berguna, yang merupakan indeks dan perlu dihilangkan.
2. Terdapat outlier yang perlu ditangani karena dapat mempengaruhi kinerja model
3. Kolom "y" dan "z" memiliki beberapa outlier dimensional dalam dataset dan perlu dihilangkan.
4. Fitur "depth" & "table" seharusnya dibatasi setelah diperiksa Plot Garis.

### Memeriksa Potensi Outlier


```python
lm = sns.lmplot(x = 'price', y= 'y', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs y', color = '#774571', fontsize = 15)
plt.show()
```


    
![png](output_26_0.png)
    



```python
lm = sns.lmplot(x = 'price', y= 'z', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs z', color = '#774571', fontsize = 15)
plt.show()
```


    
![png](output_27_0.png)
    



```python
lm = sns.lmplot(x = 'price', y= 'depth', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs depth', color = '#774571', fontsize = 15)
plt.show()
```


    
![png](output_28_0.png)
    



```python
lm = sns.lmplot(x = 'price', y= 'table', data = df, scatter_kws = {'color': "#FED8B1"}, line_kws = {'color' : '#4e4c39'})
plt.title('Plot Garis Price vs table', color = '#774571', fontsize = 15)
plt.show()
```


    
![png](output_29_0.png)
    


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




    
![png](output_40_1.png)
    


Catatan:

Fitur "carat", "x", "y", "z" memiliki korelasi yang tinggi dengan variabel target kita, yaitu harga.

Fitur "cut", "clarity", "depth" memiliki korelasi yang sangat rendah (<|0.1|) sehingga mungkin dapat dihapus, meskipun karena hanya ada beberapa fitur yang dipilih, kita tidak akan melakukannya.

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


```python
cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, x_train,y_train,scoring="neg_root_mean_squared_error", cv=12)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipeline_dict[i], -1 * cv_score.mean()))
```

    LinearRegression: 1383.854012 
    Lasso: 1366.991298 
    DecisionTree: 739.289291 
    RandomForest: 548.841032 
    KNeighbors: 816.559263 
    XGBRegressor: 548.346850 
    

# 5. EVALUATION


```python
# Prediksi model pada data pengujian dengan XGBClassifier yang memberikan kita RMSE paling sedikit
pred = pipeline_xgb.predict(x_test)
print("R^2:",metrics.r2_score(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
```

    R^2: 0.9821291192884176
    Adjusted R^2: 0.9821141881775372
    




```python

```
