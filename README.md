# semantic-hashing
Implementacion en Python de Semantic Hashing.

## Instrucciones
Para entrenar la DBN y generar las cadenas binarias se realiza:
```python
import semantic_hashing as sh
W = sh.fit(train_data)
hashs = sh.transform(test_data,W)
```

## Métodos
### fit()
Entrena la DBN según los tfs, la cantidad de capas ocultas, tamaño de los batchs. Retorna una lista con los pesos de la red.
```python
fit(tfs, hidden_layer=[500,500], output_layer=128, maxepoch=50,
    lr_w = 0.1, lr_vb = 0.1, lr_hb = 0.1, weightcost = 0.0002, momentum = 0.9,
    pretrain_size_batch=100, finetuning_size_batch=1000)
```
***
* tfs: Numpy Array de tfs.
* hidden_layer: Lista con tamaño de las capas ocultas.
* output_layer: Tamaño de capa de salida.
* maxepoch: Total epochs en pre-entrenamiento y finetuning.
* lr_w: learning_rate para los pesos.
* lr_vb: learning_rate para la capa visible en pre-entrenamiento.
* lr_hb: learning_rate para la capa oculta en pre-entrenamiento.
* weightcost: weight decay en pre-entrenamiento.
* momentum: momentum en pre-entrenamiento.
* pretrain_size_batch: Tamaño de cada batch en pre-entrenamiento.
* finetuning_size_batch: Tamaño de cada batch en fine-tuning.
***

### transform()
Transforma los tfs en cadenas binarias.
```python
transform(tfs,W)
```
***
* tfs: Numpy Array de tfs.
* W: Lista con los pesos de la red.
***

Copyright: see http://www.cs.toronto.edu/~hinton/code/README.txt
