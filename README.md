# Trash - директория

Директория содержит файлы для оценки

[LinkNet](Python/LinkNetBoba.py) - реализация версии LinkNet на Python для [Keras](https://keras.io/backend/). Реализация носит экспериментальный характер и свободна для распространения. Я буду признателен за замечания по этой реализации.

LinkNet описан в публикации [arxiv:1707.03718](https://arxiv.org/pdf/1707.03718.pdf) 
 
*Abhishek Chaurasia, Eugenio Culurciello*

*LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation*

Встреченные проблемы при реализации:
- Несовпадение размерности при суммировании в блоке Encoder по последней размерности.
Проблема решена аналогично решению для блока в сети Resnet
- Последний UpSampling приводит к неправильному результату по размерности. Поэтому он закомментирован.
- Некоторые варианты входных размерностей приводят к несовпадению размерностей при суммировании в блоке Levels. Например, 640*360
Как с этим бороться я пока не знаю.

В целом версия рабочая и свободна для использования и модификаций. За конструктивные замечания буду признателен.

2017-11-28
