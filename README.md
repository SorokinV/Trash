# Trash - директория

Директория содержит файлы для оценки

[LinkNet](Python/LinkNetBoba.py) - реализация версии LinkNet на Python для [Keras](https://keras.io/backend/). Реализация носит экспериментальный характер и свободна для распространения. Я буду признателен за замечания по этой реализации.

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
