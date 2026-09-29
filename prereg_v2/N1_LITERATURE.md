# N1 — литература и ссылки (PREREG_v2, 2026-09-28)

Метки: [VERIFIED-tool] = найдено в OpenAlex/Crossref (scholarly-lookup) или на странице (WebFetch) в этой сессии;
[TITLE-ONLY] = найдено по названию, аннотация не читалась; [UNVERIFIED] = проверить не удалось.
Инструмент `arxiv.get_abstract` НЕ годится как проверка существования: заведомо существующий ID (2410.06686) он тоже вернул как «не найдено».

## 1. Ссылки препринта v1 (14 позиций)

| # | Ссылка в статье | Результат проверки |
|---|---|---|
| [1] | Lu & Raz, PNAS 114, 5083 (2017) | найдено, DOI 10.1073/pnas.1701264114 [VERIFIED-tool] |
| [2] | Klich, Raz, Hirschberg, Vucelja, PRX 9, 021060 (2019) | найдено, DOI 10.1103/physrevx.9.021060 [VERIFIED-tool] |
| [3] | Kumar & Bechhoefer, Nature 584, 64 (2020) | найдено, DOI 10.1038/s41586-020-2560-x [VERIFIED-tool] |
| [4] | Mpemba & Osborne, Phys. Educ. 4, 172 (1969) | Crossref-поиск ошибка/не найдено; работа доинтернетной эпохи [UNVERIFIED] |
| [5] | Lasanta et al., PRL 119, 148001 (2017) | найдено, DOI 10.1103/physrevlett.119.148001 [VERIFIED-tool] |
| [6] | Gal & Raz, PRL 124, 060602 (2020) | найдено, DOI 10.1103/physrevlett.124.060602 [VERIFIED-tool] |
| [7] | Baity-Jesi et al., PNAS 116, 15350 (2019) | найдено, DOI 10.1073/pnas.1819803116; страница не сверялась [VERIFIED-tool, частично] |
| [8] | Kramers, Physica 7, 284 (1940) | найдено, DOI 10.1016/s0031-8914(40)90098-2 [VERIFIED-tool] |
| [9] | Lapolla & Godec, PRL 125, 110602 (2020) | найдено, DOI 10.1103/physrevlett.125.110602; **существует erratum** PRL 128, 229901 (2022), DOI 10.1103/physrevlett.128.229901 — в списке отсутствует [VERIFIED-tool] |
| [10] | Vu & Hayakawa, PRL 134, 107101 (2025) | найдено, DOI 10.1103/physrevlett.134.107101 [VERIFIED-tool] |
| [11] | Liu & Hu, arXiv:2507.04206 | найдено по названию: «Mpemba Effect in Large-Language Model Training Dynamics: A Minimal Analysis of the Valley-River model» (реальное название длиннее, чем в статье); ID не подтверждён, авторы не показаны [TITLE-ONLY]; **подтверждено 2026-09-29** страницей arxiv.org/abs/2507.04206: авторы Sibei Liu и Zhijian Hu (в v1 «J. Liu» — инициал неверен), подано 2025-07-06 [VERIFIED-tool, WebFetch] |
| [12] | Teza, Bechhoefer et al., arXiv:2502.01758 | ID подтверждён (DOI 10.48550/arxiv.2502.01758); **название в статье неверно** («a review of the…» отсутствует: реальное — «Speedups in nonequilibrium thermal relaxation: Mpemba and related effects»); работа уже опубликована в Physics Reports, DOI 10.1016/j.physrep.2025.10.009 — цитировать журнальную версию [VERIFIED-tool] |
| [13] | Chattopadhyay, Santos, Misra, arXiv:2601.05046 | ID и название подтверждены (DOI 10.48550/arxiv.2601.05046); авторы не показаны [VERIFIED-tool, частично] |
| [14] | Summer et al., PRX 16, 011065 (2026), DOI 10.1103/rbt4-psfd | DOI и название подтверждены; том/номер статьи не сверялись [VERIFIED-tool, частично] |

**Нумерация в тексте v1 нарушена** (прочитано в `paper/preprint_v1.md`, введение): «spin systems [5], colloidal particles [6],
granular gases [7], clathrate hydrates [8]» — но [5] — гранулярные жидкости, [6] — Gal & Raz (предохлаждение), [7] — спиновые
стёкла, [8] — Kramers (1940); клатратных гидратов в списке нет вовсе. Все четыре привязки неверны [VERIFIED-read].

## 2. Пересечение с опубликованной работой (новизна)

Найденные по названиям в реестрах; две ближайшие аннотации прочитаны на arXiv (WebFetch, [VERIFIED-tool]):

- **Hayakawa & Takada, «Mpemba effect in a two-dimensional bistable potential», arXiv:2603.24148 (25.03.2026).** Аналитически решаемая
  модель Мпемба для передемпфированного Ланжевена в 2D радиально-симметричном бистабильном потенциале; оператор Фоккера–Планка
  сводится к задаче Шрёдингера; явные амплитуды мод; коэффициент самой медленной моды; условия пересечения KL. Тот же класс задач и тот же аппарат, что у нас.
- **Biswas, Prasad, Rajesh, «Mpemba effect in driven granular gases: role of distance measures», arXiv:2303.10900 (2023).** Четыре меры
  расстояния; вывод: наличие эффекта Мпемба зависит от определения меры. Центральный тезис v1 об «observable-specificity» в общей форме опубликован раньше, на другой системе.
- [TITLE-ONLY]: «Mpemba effect in a Langevin system: Population statistics, metastability, and other exact results» (J. Chem. Phys. 2023,
  10.1063/5.0155855); «The Metastable Mpemba Effect Corresponds to a Non-monotonic Temperature Dependence of Extractable Work» (Front. Phys. 2021);
  «Thermal Versus Entropic Mpemba Effect in Molecular Gases with Nonlinear Drag» (PRE 2022); «Anomalous heating in a colloidal system» (PNAS 2022);
  «Microscopic theory of Mpemba effects and a no-Mpemba theorem for monotone many-body systems» (arXiv:2410.06623);
  «Mpemba effect for a Brownian particle trapped in a single well potential» (PRE 2023); «Predicting the conditions for observing the Mpemba effect» (arXiv:2606.03445).

**Вывод N1:** слова «new / novel» для механизма, для «observable-specificity» и для самой модели неприменимы. Заметка должна цитировать Lu–Raz, Klich et al.,
Biswas et al., Hayakawa & Takada как предшествующие работы. Вклад может быть только методическим (ловушки оценки равновесия по хвостам, проверка порядка
начальных расстояний, валидация решателя) — при условии, что результат основного прогона это подтвердит.

## 3. Обновление после чтения аннотаций (2026-09-28, после первого прохода skeptic)

Прочитано по записям OpenAlex (WebFetch, [VERIFIED-tool]); полные тексты не читались.

- **Biswas, Rajesh, Pal, «Mpemba effect in a Langevin system: Population statistics, metastability, and other exact results», J. Chem. Phys. 159 (4), 2023
  (DOI 10.1063/5.0155855).** Точное спектральное разложение уравнения Фоккера–Планка в кусочно-линейном двухъямном потенциале; вывод авторов: ни метастабильность, ни
  асимметрия ландшафта не являются необходимыми или достаточными для эффекта; он определяется начальной статистикой заселённостей и выбором наблюдаемой.
  **Изменение вывода, не только ссылки:** наш результат «пересечение присутствует и при мелких барьерах (b=0.1, T=1), потому что симметричное hot не возбуждает нечётную моду» —
  по существу тот же вывод, полученный ранее. Претензии на его новизну нет; работа должна быть процитирована как основной предшественник.
- **Vu & Hayakawa, PRL 134, 107101 (2025), «Thermomajorization Mpemba effect».** Аннотация: обычная теория опирается на конкретный выбор меры расстояния, что даёт неоднозначности;
  формулировка через термомажоризацию снимает их и объединяет все монотонные меры. **Изменение claim'а:** в черновике заметки эта работа была процитирована как источник «спектрального происхождения»
  эффекта — привязка неверна; она относится к формулировке, не зависящей от меры расстояния, и обсуждается рядом с выбором KL/W1 (W1 не является f-дивергенцией).
- Проверка DOI ссылок [9]–[14] препринта v1 сведена с этой литературой в один список (раздел 1); не подтверждены [4] и ID [11] (ID [11] подтверждён позже, 2026-09-29, см. раздел 5).

## 4. Что реально прочитано (2026-09-28, после третьего прохода skeptic)

- **Аннотация прочитана:** Klich et al. 2019 (определяет «сильный» эффект: время релаксации скачком уменьшается, экспоненциально быстрее; OpenAlex); Vu & Hayakawa 2025; Biswas–Rajesh–Pal 2023; Biswas & Rajesh, PRE 108, 024131 (2023) (одноямный кусочно-линейный потенциал без других метастабильных минимумов, точное решение; эффект в широкой области параметров; оспаривает объяснение «нужен неровный ландшафт»);
  Biswas–Prasad–Rajesh 2023; Hayakawa & Takada 2026.
- **Только название и авторы:** Chétrite, Kumar, Bechhoefer, Front. Phys. 2021 (DOI 10.3389/fphy.2021.654271).
- **Только идентификаторы (DOI/название):** Lu & Raz 2017; Kumar & Bechhoefer 2020 (аннотация в OpenAlex отсутствует); Teza et al. 2025; Scharfetter–Gummel 1969; Grassmann–Taksar–Heyman 1985. Полных текстов не читалось.
- В заметке содержательные утверждения об этих работах (спектральное происхождение у Lu & Raz; коллоидная частица в потенциальном ландшафте у Kumar & Bechhoefer) помечены как «по памяти».

## 5. Ссылки [4] и [11] препринта v1 (2026-09-28)

- **[11] Liu & Hu:** запись OpenAlex W4415346113 существует, 2025 г.; название там «Mpemba Effect in Large-Language Model Training Dynamics: A Minimal Analysis of the Valley-River model»; авторы указаны как «S Liu, Zhijian Hu» (в v1: «J. Liu and Z. Hu» — инициал, вероятно, неверен); DOI и arXiv-ID в записи не показаны, так что ID 2507.04206 остаётся непроверенным. Аннотация прочитана (анализ «valley-river» ландшафтов, точка «strong point», где медленная мода исчезает).
  **Обновление 2026-09-29:** страница arxiv.org/abs/2507.04206 подтверждает ID, точное название, авторов (Sibei Liu, Zhijian Hu) и дату подачи (6 июля 2025); в аннотации — связь с расписанием learning rate WSD и «strong Mpemba point». Ссылка [11] проверена; поправка к v1: инициал «S.», не «J.». Полный текст не читался.
- **[4] Mpemba & Osborne 1969:** два запроса к реестру вернули HTTP 400; не проверено (работа доцифровой эпохи); проверить автору вручную.
