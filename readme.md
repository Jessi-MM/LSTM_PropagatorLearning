# Notas

## Sobre `class ProtonTransfer`


El constructor de la clase utiliza los siguientes valores:  
- n: Número de puntos en el grid para el método DVR
- a: Punto inicial del grid [angstroms]
- b: Punto final del grid [angstroms]
- k: Número de eigenestados incluidos para el paquete inicial de onda
- time: True o False. Determina si se utiliza un potencial dependiente del tiempo: True, o independiente del tiempo: False  
- var_random: True o False. True inicia las variables de manera aleatoria para el potencial del sistema. False solicita al usuario.
- save_dir: Nombre del directorio donde se guardarán los datos del potencial y la evolución de onda.

*Ejemplo*:  

`data = ProtonTransfer(k = 5, n = 32, a=-1.5, b=1.5, time=True, var_random=True, save_dir='data')`

### Para generar un paquete de onda inicial aleatorio   

Se utiliza el método Wavepacket_Init:

`data.Wavepacket_Init()`  

Devuelve un array de $n$ números complejos

### Para generar potenciales dependientes del tiempo:V(t) (valores aleatorios)

Se utiliza el método vector_potential:  
`data.vector_potential(t, step)`  

Devuelve un array de $n$ números reales con el potencial [au] al tiempo t en cada punto del grid.  
Actualización: Guarda los datos por cada step en la carpeta 'data/Potential/step-potential.npy'

### Para generar potenciales independientes del tiempo: V (valores aleatorios)

Se utiliza el método vector_potential:  
`data.vector_potential_TI()`  

Devuelve un array de $n$ números reales con el potencial [au] en cada punto del grid.   
actualización: Guarda el potencial 'data/Potential/step-potential.npy'

### Para generar un paquete de onda propagado al tiempo t con V(t)  

Se utiliza el método evolution_wp:  
`data.evolution_wp(t = 100, step = 1, gaussiana=False)`   

Devuelve un array de $n$ números complejos correspondientes a la evolución de un paquete inicial (`data.Wavepacket_Init()`)al tiempo t [fs], con pasos de 1 [fs].  

`data.evolution_wp(t = 100, step = 1, gaussiana=True)`  
Devuelve un array de $n$ números complejos correspondientes a la evolución de un paquete inicial gaussiano al tiempo t [fs], con pasos de 1 [fs].  

Actualización: Guarda el paquete de onda a cada step en el tiempo en: 'data/Wavepacket/step-wavepacket.npy'

### Para generar un paquete de onda propagado al tiempo t con V  

Se utiliza el método evolution_wp:  
`data.evolution_wp_TI(t = 100, step = 1)`   

Devuelve un array de $n$ números complejos correspondientes a la evolución de un paquete inicial al tiempo t [fs], con pasos de 1 [fs]. Para un potencial independiente del tiempo.   
Actualización: Guarda el paquete de onda a cada step en el tiempo en: 'data/Wavepacket/step-wavepacket.npy'   

### Guardar las variables de cada dato generado en archivos txt:
`data.get_values_Trajectory(t, step=1)`
`data.get_values_Potential()`


## Sobre `class Potential_System`  

Se utiliza para generar las funciones del modelo físico del potencial: [artículo principal](https://doi.org/10.1021/acs.jpclett.1c03117)
