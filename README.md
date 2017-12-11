# Trash

LinkNet is a version of [LinkNet](Python/LinkNetBoba.py)  in Python for [Keras](https://keras.io). The implementation is experimental and free for distribution. I will be grateful for the comments on this implementation.

LinkNet is described in the publication [arxiv:1707.03718](https://arxiv.org/pdf/1707.03718.pdf) 

*Abhishek Chaurasia, Eugenio Culurciello*

*LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation*

Implementation in Python. The implementation is unpretentious to the grid dimensions, which is specified by the transmission of shape parameters during generation. The use of all basic constructions (BatchNormalization, DropOut, Activation) is parameterized. A new operator for DropOut was used.

The implementation is built in such a way that it can be reworked into any intelligible shell, replacing the basic operators in the design.

The description problems found during the implementation:

 - Mismatch of the dimension when summing in the Encoder block by the last dimension. The problem is solved similarly to the solution for the block in the Resnet network
 - The last UpSampling leads to the wrong result in dimension. Therefore, it is commented out.
 - Some variants of input dimensions lead to a mismatch of dimensions when summing in the Levels block. For example, 640 * 360 I do not know how to deal with this.

The whole version is working and free for use and modifications. For constructive comments I will be grateful.

2017-11-28

PS. In my estimation, on a similar task, it works 4-5 times faster than the UNet implementation. It significantly reduces the requirements for GPU memory, which allows to significantly increase batch_size. The results are approximately the same.

----

[LinkNet](Python/LinkNetBoba.py) - реализация версии LinkNet на Python для [Keras](https://keras.io). Реализация носит экспериментальный характер и свободна для распространения. Я буду признателен за замечания по этой реализации.

LinkNet описан в публикации [arxiv:1707.03718](https://arxiv.org/pdf/1707.03718.pdf) 
 
*Abhishek Chaurasia, Eugenio Culurciello*

*LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation*

Реализация на Python. Реализация непритязательна к размерам сетки, которая задается передачей shape параметров при генерации.
Использование всех основных конструкций (BatchNormalization, DropOut, Activation) запараметризовано. Использован новый оператор для DropOut.

Реализация построена таким образом, что может быть переработана на любую внятную оболочку, заменой основных операторов в конструкции.

Проблемы описания, обнаруженные при реализации:

- Несовпадение размерности при суммировании в блоке Encoder по последней размерности.
Проблема решена аналогично решению для блока в сети Resnet
- Последний UpSampling приводит к неправильному результату по размерности. Поэтому он закомментирован.
- Некоторые варианты входных размерностей приводят к несовпадению размерностей при суммировании в блоке Levels. Например, 640*360
Как с этим бороться я пока не знаю.

В целом версия рабочая и свободна для использования и модификаций. За конструктивные замечания буду признателен.

2017-11-28 

PS. По моей оценке на аналогичной задаче работает в 4-5 раз быстрее, чем UNet-реализация. Существенно уменьшает требования к памяти GPU, что позволяет существенно увеличить batch_size. Результаты приблизительно одинаковые.

Keys: Keras LinkNet 1707.03781 LinkNetBoba.py
