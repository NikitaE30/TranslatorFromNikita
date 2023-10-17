# TranslatorFromNikita
## RU
### Описание
**TranslatorFromNikita** – это универсальное решение по локальному переводу текста, с использованием множества языков. Проект использует
существующее **open-source** решение [Argos-translate](https://github.com/argosopentech/argos-translate). Преимущество использования **TranslatorFromNikita**
состоит в легкой установке, отсутствии логирования и редкого подключения к интернету, простой и быстрой работе.
### Причины создания
При использовании решения [Argos-translate](https://github.com/argosopentech/argos-translate), требуется наличие [Stanza](https://github.com/stanfordnlp/stanza) (библиотеки, направленной на обработку естественного языка). В совокупности они представляют следующие недостатки:
- использование логирования;
- редкое подключение к интернету;
- большое количество неиспользуемых функций (только при использовании обычного перевода);
- непонятные пути установки моделей;
- хранение моделей в корне Python.

**TranslatorFromNikita** позволяет избавиться от подобных недостатков. При создании решения было вырезано **1.75 мб** ненужного кода. Вес кода **TranslatorFromNikita** составляет **246 кб**. Было удалено большинство зависимостей. Ни при каких условиях не будет производиться попытка скачать/отправить что-либо через интернет. Полностью автономная работа.
### Доступные языки
|       Метод перевода      |     Код     |       Метод перевода      |     Код     |       Метод перевода      |     Код     |
|---------------------------|-------------|---------------------------|-------------|---------------------------|-------------|
|  Arabic --> English       |  ar --> en  |  English --> Russian      |  en --> ru  |  English --> Hebrew       |  en --> he  |
|  Azerbaijani --> English  |  az --> en  |  English --> Slovak       |  en --> sk  |  Dutch --> English        |  nl --> en  |
|  Catalan --> English      |  ca --> en  |  English --> Swedish      |  en --> sv  |  Polish --> English       |  pl --> en  |
|  Czech --> English        |  cs --> en  |  English --> Thai         |  en --> th  |  Portuguese --> English   |  pt --> en  |
|  Danish --> English       |  da --> en  |  English --> Turkish      |  en --> tr  |  English --> Hindi        |  en --> hi  |
|  German --> English       |  de --> en  |  English --> Ukranian     |  en --> uk  |  English --> Hungarian    |  en --> hu  |
|  Greek --> English        |  el --> en  |  English --> Chinese      |  en --> zh  |  Russian --> English      |  ru --> en  |
|  English --> Arabic       |  en --> ar  |  Esperanto --> English    |  eo --> en  |  English --> Indonesian   |  en --> id  |
|  English --> Azerbaijani  |  en --> az  |  Spanish --> English      |  es --> en  |  Slovak --> English       |  sk --> en  |
|  English --> Catalan      |  en --> ca  |  Persian --> English      |  fa --> en  |  English --> Italian      |  en --> it  |
|  English --> Czech        |  en --> cs  |  Finnish --> English      |  fi --> en  |  Swedish --> English      |  sv --> en  |
|  English --> Danish       |  en --> da  |  French --> English       |  fr --> en  |  English --> Japanese     |  en --> ja  |
|  English --> German       |  en --> de  |  Irish --> English        |  ga --> en  |  Thai --> English         |  th --> en  |
|  English --> Greek        |  en --> el  |  Hebrew --> English       |  he --> en  |  English --> Korean       |  en --> ko  |
|  English --> Esperanto    |  en --> eo  |  Hindi --> English        |  hi --> en  |  Turkish --> English      |  tr --> en  |
|  English --> Spanish      |  en --> es  |  Hungarian --> English    |  hu --> en  |  Ukranian --> English     |  uk --> en  |
|  English --> Persian      |  en --> fa  |  Indonesian --> English   |  id --> en  |  Chinese --> English      |  zh --> en  |
|  English --> Finnish      |  en --> fi  |  Italian --> English      |  it --> en  |  English --> Dutch        |  en --> nl  |
|  English --> French       |  en --> fr  |  Japanese --> English     |  ja --> en  |  English --> Polish       |  en --> pl  |
|  English --> Irish        |  en --> ga  |  Korean --> English       |  ko --> en  |  English --> Portuguese   |  en --> pt  |

### Установка
Порядок действий:
```
  git clone https://github.com/NikitaE30/TranslatorFromNikita.git
  cd TranslatorFromNikita
  pip install -r requirements.txt
```
> Пожалуйста, извлеките папку **TranslatorFromNikita** из скачанного репозитория в корень Вашего основного проекта. В противном случае, Вы не сможете его импортировать.
### Использование
Перевод текста на определённый язык:
```
  import TranslatorFromNikita.translator
  if __name__ == "__main__":
    TranslatedText = TranslatorFromNikita.translator.translate("Всё работает!", "ru", "en")
    print(TranslatedText)
```
Установка модели переводчика:
```
  import TranslatorFromNikita.translator
  if __name__ == "__main__":
    TranslatorFromNikita.translator.install_from_path("<путь>.argosmodel")
```
> Скачать модели переводчика Вы можете с официального сайта [Argos-translate](https://www.argosopentech.com/argospm/index/). Или скачать **мой** архив со всеми моделями по [ссылке](https://drive.google.com/file/d/1_WNtoZ1p58L0ofyHkFZFLc5-qSXYVbC8/view?usp=sharing) и извлечь его содержимое в: **./TranslatorFromNikita/translators/**. Пароль на архив: **https://github.com/NikitaE30**
### Текущая версия
Текущая версия **TranslatorFromNikita** - 1.0.0. Это первая, полностью работоспособная версия. В ней были реализованы 
следующие функции:
- перевод текста (60 вариантов перевода);
- установка новой модели переводчика.
### Примечание
Данный проект является лишь слиянием [Argos-translate](https://github.com/argosopentech/argos-translate) и [Stanza](https://github.com/stanfordnlp/stanza). Я не претендую на звание автора и указываю прямые ссылки на авторов данных библиотек. Код проекта имеет мало схожего с основным кодом двух, выше перечисленных, проектов. Текущий код проекта не отвечает требованиям [PEP-8](https://pep8.org/) и далёк от идеала.
# TranslatorFromNikita
## EN
### Description
**TranslatorFromNikita** is a universal solution for local text translation, using a variety of languages. The project uses
the existing **open-source** solution [Argos-translate](https://github.com/argosopentech/argos-translate). The advantage of using 
**TranslatorFromNikita** is easy installation, no logging, no rare internet connection, simple and fast operation.
### Reasons for creation
When using the [Argos-translate](https://github.com/argosopentech/argos-translate)solution, requires availability [Stanza](https://github.com/stanfordnlp/stanza) (libraries aimed at natural language processing). Together , they present the following disadvantages:
- using logging;
- rare internet connection;
- a large number of unused functions (only when using regular translation);
- unclear ways to install models;
- storing models in the Python root.

**TranslatorFromNikita** allows you to get rid of such shortcomings. When creating the solution, **1.75 mb** of unnecessary code was cut out. The weight of the code **TranslatorFromNikita** is **246 kb**. Most of the dependencies have been removed. Under no circumstances will an attempt be made to download/send anything via the Internet. Fully autonomous operation.
### Available languages
|    Translation methods    |    Codes    |    Translation methods    |    Codes    |    Translation methods    |    Codes    |
|---------------------------|-------------|---------------------------|-------------|---------------------------|-------------|
|  Arabic --> English       |  ar --> en  |  English --> Russian      |  en --> ru  |  English --> Hebrew       |  en --> he  |
|  Azerbaijani --> English  |  az --> en  |  English --> Slovak       |  en --> sk  |  Dutch --> English        |  nl --> en  |
|  Catalan --> English      |  ca --> en  |  English --> Swedish      |  en --> sv  |  Polish --> English       |  pl --> en  |
|  Czech --> English        |  cs --> en  |  English --> Thai         |  en --> th  |  Portuguese --> English   |  pt --> en  |
|  Danish --> English       |  da --> en  |  English --> Turkish      |  en --> tr  |  English --> Hindi        |  en --> hi  |
|  German --> English       |  de --> en  |  English --> Ukranian     |  en --> uk  |  English --> Hungarian    |  en --> hu  |
|  Greek --> English        |  el --> en  |  English --> Chinese      |  en --> zh  |  Russian --> English      |  ru --> en  |
|  English --> Arabic       |  en --> ar  |  Esperanto --> English    |  eo --> en  |  English --> Indonesian   |  en --> id  |
|  English --> Azerbaijani  |  en --> az  |  Spanish --> English      |  es --> en  |  Slovak --> English       |  sk --> en  |
|  English --> Catalan      |  en --> ca  |  Persian --> English      |  fa --> en  |  English --> Italian      |  en --> it  |
|  English --> Czech        |  en --> cs  |  Finnish --> English      |  fi --> en  |  Swedish --> English      |  sv --> en  |
|  English --> Danish       |  en --> da  |  French --> English       |  fr --> en  |  English --> Japanese     |  en --> ja  |
|  English --> German       |  en --> de  |  Irish --> English        |  ga --> en  |  Thai --> English         |  th --> en  |
|  English --> Greek        |  en --> el  |  Hebrew --> English       |  he --> en  |  English --> Korean       |  en --> ko  |
|  English --> Esperanto    |  en --> eo  |  Hindi --> English        |  hi --> en  |  Turkish --> English      |  tr --> en  |
|  English --> Spanish      |  en --> es  |  Hungarian --> English    |  hu --> en  |  Ukranian --> English     |  uk --> en  |
|  English --> Persian      |  en --> fa  |  Indonesian --> English   |  id --> en  |  Chinese --> English      |  zh --> en  |
|  English --> Finnish      |  en --> fi  |  Italian --> English      |  it --> en  |  English --> Dutch        |  en --> nl  |
|  English --> French       |  en --> fr  |  Japanese --> English     |  ja --> en  |  English --> Polish       |  en --> pl  |
|  English --> Irish        |  en --> ga  |  Korean --> English       |  ko --> en  |  English --> Portuguese   |  en --> pt  |

### Installation
Procedure of actions:
```
  git clone https://github.com/NikitaE30/TranslatorFromNikita.git
  cd TranslatorFromNikita
  pip install -r requirements.txt
```
> Please extract the **TranslatorFromNikita** folder from the downloaded repository to the root of your main project. Otherwise, you won't be able to import it.
### Using
Translating text into a specific language:
```
  import TranslatorFromNikita.translator
  if __name__ == "__main__":
    TranslatedText = TranslatorFromNikita.translator.translate("Всё работает!", "ru", "en")
    print(TranslatedText)
```
Installing the Translator model:
```
  import TranslatorFromNikita.translator
  if __name__ == "__main__":
    TranslatorFromNikita.translator.install_from_path("<path>.argosmodel")
```
> You can download the translator models from the official website [Argos-translate](https://www.argosopentech.com/argospm/index/). Or download **my** archive with all models from [link](https://drive.google.com/file/d/1_WNtoZ1p58L0ofyHkFZFLc5-qSXYVbC8/view?usp=sharing) and extract its contents to: **./TranslatorFromNikita/translators/**. Password for the archive: **https://github.com/NikitaE30**
### Current version
Current version **TranslatorFromNikita** - 1.0.0. This is the first fully functional version. The following functions were implemented in it:
- text translation (60 translation options);
- installing a new translator model.
### Note
This project is only a merger of [Argos-translate](https://github.com/argosopentech/argos-translate ) and [Stanza](https://github.com/stanfordnlp/stanza). I do not claim to be the author and indicate direct links to the authors of these libraries. The project code has little in common with the main code of the two projects listed above. The current project code does not meet the requirements of [PEP8](http://pep8.org/) and far from ideal.