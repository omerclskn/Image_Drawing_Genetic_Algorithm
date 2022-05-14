
import numpy as np
from matplotlib import pyplot as plt
import random

# cizilecek resim matrisi
target_matrix = np.array(
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)

# hareketler için kullanacağımız matris
hareket = [[0, 1],  # asagi 0
           [-1, 1],  # sol asagi 1
           [-1, 0],  # sol 2
           [-1, -1],  # sol yukari 3
           [0, -1],  # yukari 4
           [1, -1],  # sağ yukari 5
           [1, 0],  # sağ 6
           [1, 1]]  # sağ asagi 7

area_x = len(target_matrix)  # resim matrisinin 1. boyutu
area_y = len(target_matrix)  # resim matrisinin 2. boyutu
Pop_Size = 1000  # popülasyon sayısı
mu = 0.05   # her hareket için mutasyon olasılığı
Gen_Size = 201   # jenerasyon sayısı
elit_num = 1 # elitizm yapılacak birey sayısı

avg_instance = [] # bir jenerasyondaki bireylerin ortalaması
best_instance = [] # bir jenerasyondaki en iyi birey
all_time_best = [] # bütün jenerasyonların içerisinde en iyi birey

def mutation(ind, mup = mu): # mutasyon islemi
  if random.randint(0,1) < mup:
    ind[random.randint(0,len(ind)-1)] = random.randint(0, 7) # random secilen yere yeni random hareket ataması
  return ind

def get_filled(matrix): # matriste dolu yerlerin sayısının alınması
  sum = 0
  for i in range(area_x):
    for j in range(area_y):
      if matrix[i][j] == 1: # hücre ici 1 ise sum degerinin arttırılması
        sum += 1
  return sum

lenS = get_filled(target_matrix) # bireylerin hareket sayısının bulunması, dolu hücre sayısı kadar hareket etmeli

def get_sim_fitness(generated): # benzerlik degeri gezilmesi gereken noktalar baz alınarak hesaplanmaktadır
    jac11 = 0
    jac00 = 0

    for i in range(area_x):
      for j in range(area_y):
        if target_matrix[i][j] == generated[i][j] and target_matrix[i][j] == 1: # iki matrisin karsılıklı 1 oldukları nokta
          jac11 += 1
        if target_matrix[i][j] == generated[i][j] and target_matrix[i][j] == 0: # iki matrisin karsılıklı 0 oldukları nokta
          jac00 += 1
    return (jac11)/(area_x*area_y-jac00)

def get_angle_fitness(ind): # acı degerlerinin hesaplanması
  if len(ind) == 0:
    return 1
  sum = 0
  for m in range(lenS - 1):# fitness fonksiyonunu hesaplamamız için her iki ardışık hareket arasındaki açı farkını buluyoruz.
            instance_next = ind[m + 1]
            instance_curr = ind[m]
            sum = sum + (abs(4 - (abs(instance_next - instance_curr))))*45 # iki yön arasındaki açıyı bulup tüm dizi içerisinde topluyoruz
  return sum

def draw(ind):
  cur_x = len(target_matrix) - 1
  cur_y = 0
  new_matrix = np.zeros([len(target_matrix), len(target_matrix)], dtype = int) # resim matrisi boyutunda bos matris
  new_matrix[cur_x][cur_y] = 1
  for i in range(len(ind)): # hareket dizisinin matristeki karsılıklarına gore cizilmesi
     cur_x += hareket[ind[i]][0] # x eksenindeki hareketler
     cur_y += hareket[ind[i]][1] # y eksenindeki hareketler
     new_matrix[cur_x][cur_y] = 1 # lokasyonun isaretlenmesi
  return new_matrix

def drawByUnit(ind):
  cur_x = len(target_matrix) - 1
  cur_y = 0
  new_matrix = np.zeros([len(target_matrix), len(target_matrix)], dtype = int) # resim matrisi boyutunda bos matris
  new_matrix[cur_x][cur_y] = 1
  for i in range(len(ind)): # hareket dizisinin matristeki karsılıklarına gore cizilmesi
     plt.clf()
     cur_x += hareket[ind[i]][0] # x eksenindeki hareketler
     cur_y += hareket[ind[i]][1] # y eksenindeki hareketler
     new_matrix[cur_x][cur_y] = 1 # lokasyonun isaretlenmesi
     plt.imshow(new_matrix)
     plt.show()

def check_index(cur_x_arr, cur_y_arr, cur_x, cur_y, temp, iter):
  if iter > 15: # eger sıkısma olusursa mecbur kabul edilmeli
    return False
  cur_x_indexes =[]
  cur_y_indexes =[]

  for i in range(len(cur_x_arr)): # x eksenindeki değerlerin indisleri
    if (cur_x + hareket[temp][0]) == cur_x_arr[i]:
      cur_x_indexes.append(i)

  for j in range(len(cur_y_arr)): # y eksenindeki değerlerin indisleri
    if (cur_y + hareket[temp][1]) == cur_y_arr[j]:
      cur_y_indexes.append(j)


  for k in range(len(cur_x_indexes)): # eger indisler esitse aynı noktaya gidilmis demektir istenmeyen hareket
    for m in range(len(cur_y_indexes)):
      if cur_x_indexes[k] == cur_y_indexes[m]:
        return True

  return False

def crossover(a, b, cross):  #tek noktadan crossover yapmamızı sağlayan fonksiyon
    arr = np.concatenate([a[:cross], b[cross:]])
    np_arr = np.array(arr)
    np_arr = np_arr.astype(int)
    return np_arr

def normalize_fitnesses(fitness_sim, fitness_angle): # fitness degerlerinin normalize edilip birlestirilmesi
  normalize_fitness_sim_arr = fitness_sim / np.sum(fitness_sim)
  normalize_fitness_angle_arr = fitness_angle / np.sum(fitness_angle)

  total_fitness = normalize_fitness_sim_arr + (1-normalize_fitness_angle_arr)
  normalize_total_fitness = total_fitness / np.sum(total_fitness)
  return normalize_total_fitness, total_fitness

def selection(cur_gen, fitness_sim_arr, fitness_angle_arr):

  normalize_fitness, total_fitness = normalize_fitnesses(fitness_sim_arr, fitness_angle_arr) 

  indexes = np.argsort(normalize_fitness) # index sort
  indexes = indexes[::-1] # buyukten kucuge

  selection = np.zeros(Pop_Size, dtype = int)  #bireylerin iyiliklerine göre ağırlıklarını içinde tutacak olan selection fonksiyonu
  for l in range(Pop_Size):
    selection[indexes[l]] = Pop_Size - l # büyükten küçüğe sıralı index değerlerini selectionun içerisine yerleştiriyoruz
  selection = selection / np.sum(selection) # selectionları normalize ediyoruz. böylece ağırlıklarına göre seçim yapabileceğiz

  chosen = np.random.choice(Pop_Size, Pop_Size, replace=True, p=selection)  # choice fonksiyonundan yararlanarak rulet tekeri yöntemiyle seçim yapıyoruz

  selected = []
  for k in range(elit_num): # en iyi bireyin mutasyonlu ve mutasyonsuz halinin alınması
    mutated = check_random_hareket(mutation(cur_gen[indexes[k]]))
    selected.append(mutated)
    selected.append(cur_gen[indexes[k]])

  avg_instance.append(np.mean(fitness_sim_arr))  # bireylerin ortalamasını alıyoruz
  best_instance.append(fitness_sim_arr[indexes[0]])  # jenerasyonun en iyi bireyini bir diziye atıyoruz

  cross = int(np.random.rand(1)*(lenS)+2)  # crossover yapılacak noktayı random olarak seçiyoruz

  for i in range(Pop_Size-elit_num): # crossover ve mutasyon islemleri random olarak yapılıyor
    temp = mutation(crossover(cur_gen[chosen[random.randint(0,Pop_Size-1)]], cur_gen[chosen[random.randint(0,Pop_Size-1)]], lenS))
    selected.append(check_random_hareket(temp))
  return selected

def check_random_hareket(ind):
  cur_x = len(target_matrix) - 1
  cur_y = 0
  cur_x_arr = [] # gidilen noktaları tutmak icin x ve y arrayleri olusturuyoruz
  cur_y_arr=[]
  cur_x_arr.append(cur_x)
  cur_y_arr.append(cur_y)
  iter = 0 # geriye donme durumunu kontrol edebilmek icin tutulan degisken
  for i in range(len(ind)):
    temp = ind[i]
    # hareketlerin tasma olusturması kontrolü ve gidilen yere bir daha gidilmemesi için check index fonksiyonu
    while cur_x + hareket[temp][0] < 0 or cur_y + hareket[temp][1] < 0 or cur_x + hareket[temp][0] > (area_x - 1) or cur_y + hareket[temp][1] > (area_y - 1) or ((cur_y + hareket[temp][1] in cur_y_arr) and (cur_x + hareket[temp][0] in cur_x_arr) and check_index(cur_x_arr, cur_y_arr, cur_x, cur_y, temp, iter) ):
        temp = random.randint(0, 7)
        iter += 1 # her bir yeni uretilen harekette degerin arttırılması
    ind[i] = temp
    cur_x = cur_x + hareket[ind[i]][0] # hareketin işlenip yeni lokasyonun elde edilmesi
    cur_y = cur_y + hareket[ind[i]][1]
    cur_x_arr.append(cur_x) # lokasyonun diziye atılması
    cur_y_arr.append(cur_y)
    iter = 0
  return ind

def init_first_gen(): # ilk jenerasyonun olusturulması
  pop = []
  for i in range(Pop_Size):
    temp = []
    for j in range(lenS):
      temp.append(random.randint(0, 7)) # ilk jenerasyon icin random hareketlerin uretilmesi
    pop.append(temp)
  for i in range(Pop_Size):
    pop[i] = check_random_hareket(pop[i]) # uretilen hareketlerin kontrol edilmesi
  return pop  

generation = init_first_gen() # jenerasyonları içeren değişken
fitness_sim = [] # benzerlik değerlerinin tutulduğu matris
fitness_angle = [] # açı değerlerinin tutulduğu matris

# genel for
for i in range(Gen_Size):
  for j in range(Pop_Size):
    ind = generation[j]
    draw_matrix = draw(ind)
    # fitness fonksiyonlarının hesaplanıp dizi icerisinde saklanması
    fitness_sim.append(get_sim_fitness(draw_matrix))
    fitness_angle.append(get_angle_fitness(ind))

  selected = selection(generation, fitness_sim, fitness_angle) # en iyi bireyin tutulması
  if (get_sim_fitness(draw(all_time_best))) < (get_sim_fitness(draw(selected[0]))):
      all_time_best = selected[0]

  generation = selected
  fitness_sim = []
  fitness_angle = []
  if i % 20 == 0: # her 10 jenerasyonda bir hareketin gosterilmesi
    best = selected[0]
    label_fitness = round(get_sim_fitness(draw(best)), 2)," benzerlik degeri",get_angle_fitness(best)," açı değeri"
    label_top = i,". jenerasyon "
    plt.plot(best_instance)
    plt.plot(avg_instance)
    plt.figure()
    fig, axarr = plt.subplots(1,2) 

    fig.suptitle(label_top, fontsize=16)
    axarr[0].imshow(draw(best))
    axarr[0].set_title(label_fitness)
    axarr[1].imshow(target_matrix)
    axarr[1].set_title("Target Image")
    
    plt.show()

drawByUnit(all_time_best)
# programın sonunda en iyi birey gosterilir
label2 = "This is the best of all generations ",round(get_sim_fitness(draw(all_time_best)), 2)," benzerlik degeri",get_angle_fitness(all_time_best)," açı değeri"
plt.title(label2)
plt.imshow(draw(all_time_best))
plt.show()