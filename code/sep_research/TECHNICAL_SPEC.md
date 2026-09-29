# sep_research — техническое задание

Редакция: 0.2, после согласования пользователем двух логических агентов, исходной reward, pilot-first подхода и additive-first реализации. Добавлены два варианта PV: локальная MLP и MLP-головы с графовым представлением. Исходная ветка `multiagentenv`, commit `0809e2361953bdebabfa60420a9594e3522156fc`.

**Нормативность.** «Должно» означает требование к будущей реализации, а не уже существующее свойство проекта. **FIXED** — исходные ограничения и согласованные решения; **PROPOSED** — конкретизация дизайна, фиксируемая до main training; **OPTIONAL** — за пределами основной матрицы. Изменения/дополнения коду разрешены, но приоритет — новые компоненты и сохранение воспроизводимости прошлого исследования. Текущий запрос выполняется только как корректировка документации. Актуальный статус решений: [DECISIONS.md](DECISIONS.md).

## 1. Цель, исследовательский объект и границы

### 1.1. Цель

Проверить, улучшает ли прямое обучаемое маневрирование одного приоритетного транспортного средства (PV), одновременно обучаемое с кооперативной политикой окружающих AV, приоритетную мобильность по сравнению с исходной схемой, где обучаются только AV, а PV следует native Revised S-NFS.

Под **лавированием** в основном эксперименте понимаются только дискретные запросы перехода на соседнюю полосу. Не непрерывный руль, не ускорение, не экстренное торможение, не многополосный прыжок, не выбор маршрута.

### 1.2. Что именно сохраняется

**FIXED:** физика и параметры финальной локальной статьи `paper/main.tex`; один PV; native longitudinal dynamics; кольцо; состав HDV/AV/PV; PV-centred AV window; hard masks; node-factorized PPO для AV; сравнимые сценарии и метрики; раздельные validation/test.

**FIXED:** две независимо параметризованные обучаемые политики, обучающиеся совместно. Нельзя выдать за это две головы одной общей сети, sequential training «сначала AV, потом PV» или frozen AV + обучаемый PV как единственный основной метод.

**FIXED, согласовано:** два постоянных логических RLlib-agent — PV и AV-координатор. Физические AV по-прежнему являются динамическими узлами, а не отдельными RLlib-trajectories. В каждом эксперименте ровно две policy; сравнение двух архитектур PV означает разные обучающие runs, не три policy в одной среде.

**FIXED, расширенный scope:** сравнить `pv_encoder=local_mlp` (L) и `pv_encoder=graph_mlp` (G). У обоих PV отдельные MLP actor/value heads; G получает локальные PV features плюс graph embedding того же типа, что у AV. **PROPOSED конкретизация «такой же»:** одинаковые raw graph/feature schema и архитектура encoder, но самостоятельные параметры PV; shared learned AV/PV encoder не входит в эти два варианта.

**FIXED:** два обучаемых типа не исключают HDV из симуляции. Доля AV относится только к non-PV автомобилям; при f=0.90/0.95 остальные — native HDV.

### 1.3. Вне основного scope

Не добавлять: speed-control, новые модели car following, светофоры, маршрутизацию, открытые границы, несколько PV, автобусы, V2X-задержки, online learning на test, RNN/history, attention/heterogeneous GNN, off-policy replay, Optuna search, constrained-RL reward, distributed cloud orchestration. Эти механизмы вводят новые независимые факторы и не требуются для проверки исходного вопроса.

## 2. Зафиксированная среда статьи

| Параметр | Значение / контракт |
|---|---|
| Топология | periodic `RingTopology` |
| Полосы / длина | M=4, L=1000; lane 0 — крайняя левая |
| Единицы | клетки, шаги; без пересчёта в км/ч и секунды |
| Длина всех ТС | 1 клетка |
| Число ТС | N=floor(M L rho), включая PV |
| Скорость HDV и AV | максимум 5 клеток/шаг |
| Скорость PV | максимум 6 клеток/шаг |
| G, q, r, S | 15, 0.99, 0.99, 2 |
| P1, P2, P3, P4 | 0.999, 0.99, 0.98, 0.01; это persistence, не braking probabilities |
| Native p_lane_change | 0.5 |
| PV assignment | один существующий автомобиль после all-HDV warm-up |
| AV assignment | floor(f (N−1)) non-PV, seed-determined nested subsets |
| AV actuation window | 30 клеток впереди, 10 позади PV, все 4 полосы |
| Sensor distance | 60 от ego-автомобиля для локальных признаков |
| AV capacity | 4 (30+10+1)=164 padded slots; PV не занимает AV-slot |
| AV cooldown | 1 шаг после выполненного non-stay override |
| Rule baseline detection/cooldown | 30 / 1 |
| Warm-up | 1000 native all-HDV шагов |
| Post-assignment burn-in | 0 |
| Training episode | 1000 controlled transitions |
| Evaluation horizon | 2000 controlled transitions |
| Train conditions | rho={0.10,0.20,0.30,0.40}, f={0.90,0.95,1.00} |
| Test conditions | rho={0.05,0.10,…,0.90}, те же три f |
| Backend | optimized для больших запусков; reference — semantic oracle |

Источник чисел: `paper/main.tex:891–1010`, `paper/data/rl_final/graph/run_config.json`, `paper/data/rl_dense/final_dense_architectures_metadata.json`. Старые параметры из README/MVP, `RevSNFS.md` и планов не подставлять автоматически.

Обновление: native/explicit lateral phase → acceleration → stochastic slow-to-start/look-ahead/perspective → state-dependent random braking → collision-safe velocity relaxation → synchronous positions. Порядок ветвлений, strict inequalities, число RNG draws и восстановление lane-change flags берутся из текущего reference/control path, не реализуются заново по сокращённому описанию статьи.

## 3. Формализация двухполитичной задачи

Полное состояние X_t включает массивы `TrafficState`, RNG состояния, cooldown по физическим vehicle_id, episode counters, приватный slot mapping; для reset также condition cycle и warm-up cache.

- p — постоянный физический ID одного PV в эпизоде.
- A_t — AV внутри PV-centred окна в X_t.
- O_AV(X_t) — наблюдение статьи.
- O_PV(X_t) — новое ego-наблюдение PV.
- a_AV,t — padded вектор действий; a_PV,t — scalar categorical action.
- theta_AV и theta_PV — непересекающиеся наборы обучаемых параметров.

Совместная политика при независимом sampling условно на наблюдениях:

`pi(a_PV,a_AV|X) = pi_PV(a_PV|O_PV) * product_{i in A} pi_AV,i(a_i|O_AV)`.

Оба решения относятся к одному X_t; X_(t+1)=F(X_t,a_PV,t,a_AV,t,xi_t) вычисляется ровно один раз. Отдельное выполнение optimizer steps не означает отдельного физического хода агентов.

Это кооперативная двухполитичная partially observed stochastic game. Несмотря на формулировку «Markov snapshot» в старых документах, локальное наблюдение не содержит полный поток и RNG, и полную наблюдаемость гарантировать нельзя. Отсутствие RNN наследуется от статьи, а не доказывает достаточность наблюдения.

## 4. Владение состоянием и роли

### 4.1. Роли симулятора не равны RLlib-agent IDs

Основной режим сохраняет:

- PV: `veh_type=PRIORITY_VEH_TYPE=3`, `controlled=False`;
- AV: `veh_type=AV_VEH_TYPE=1`, `controlled=False`;
- HDV: `veh_type=HDV_VEH_TYPE=0`, `controlled=False`.

PV получает управление через `TrafficSimulator.step_lateral_overrides`, а не через legacy controlled flags. `high_speed_vehicle_mask = state.controlled | priority_vehicle_mask` уже даёт PV vmax=6. Установка `controlled=True` AV может неожиданно поднять ему vmax и подменить физику.

Не временно перекрашивать PV в AV для построения его наблюдения; не добавлять PV в AV graph; не менять `vehicle_id` между шагами. Подмена типа портит masks, nearest-HDV признаки, priority metrics и скорость.

### 4.2. Единственный владелец transition

Новый wrapper должен владеть одним `TrafficSimulator`. Нельзя вызвать `CentralizedPvEnv.step(a_AV)` и затем отдельно `sim.step` для PV: это два longitudinal updates, разные reward timing и другой эксперимент.

Переиспользовать построение AV graph, assignment, reward helper, backend dispatch и физические метрики. Реализация разрешена пользователем с приоритетом обратной совместимости: новый orchestration layer/additional adapters предпочтительнее изменения старого `CentralizedPvEnv`. Небольшое локальное дублирование разрешено, если иначе пришлось бы менять проверенный старый path; оно требует parity tests. Общий helper извлекать из существующего source только как крайний обоснованный вариант, не ради косметического DRY. Не копировать всё физическое ядро.

## 5. RLlib environment API

**PROPOSED:** `SepResearchEnv(MultiAgentEnv)` на classic RLlib stack; две постоянные сущности:

| agent_id | policy_id | Obs | Action |
|---|---|---|---|
| `pv` | `pv_policy` | Dict PV schema | `Discrete(3)` |
| `av_coordinator` | `av_policy` | Dict graph schema | `MultiDiscrete([3]*164)` |

`possible_agents`, текущие `agents`, per-agent spaces и getters должны соответствовать зафиксированной версии RLlib. Нельзя обозначить Python dict как полноценный `spaces.Dict`, не проверив API.

### 5.1. Reset

1. Инициализировать RNG по base seed, worker_index, vector_index, не теряя EnvContext при преобразовании в dict.
2. Выбрать density×fraction из shuffled полного 12-condition цикла; при исчерпании перемешать заново.
3. Получить all-HDV warm state: training cache — один snapshot на density на worker, как в статье; evaluation — независимый warm state на replication.
4. Копировать snapshot; не мутировать cache общими numpy views.
5. Назначить один PV, потом nested AV subset. Позиции, скорости, полосы не меняются.
6. Инициализировать отдельный measurement RNG, сбросить cooldown/счётчики.
7. Построить **оба** observation из одного X_0, сохранить AV slot mapping.
8. Вернуть `(observations, infos)` с одинаковыми agent keys.

Порядок выше описывает логические обязанности. Точный порядок RNG наследуется от `CentralizedPvEnv.reset`: после инициализации np_random сначала получить пять option seeds (scenario, warmup, assignment, AV selection, measurement), затем вызвать `_sample_condition`. Перестановка этих операций меняет permutation и сценарии даже при одинаковом root seed, поэтому недопустима для legacy parity.

Явный `reset(seed=s)` должен воспроизводить начальные состояния, condition order и дальнейшую последовательность reset. Неявный reset не должен бесконечно повторять первый эпизод. Повторяемость cache регулируется существующим контрактом, не новой скрытой эвристикой.

### 5.2. Step

В нормальном joint режиме `step` принимает ровно `{pv: scalar, av_coordinator: vector}`. Пропущенный/неизвестный ключ, неверная форма/dtype/range — явная ошибка до мутации состояния и RNG. При evaluation frozen/native режимы задаются конфигурацией controller composition, не трактовкой случайно пропавшего действия.

Возврат: `(obs, rewards, terminateds, truncateds, infos)`.

- До horizon: оба terminated=False, truncated=False, `__all__=False`.
- На последнем шаге: обоим truncated=True, terminated=False, соответствующий `__all__`.
- Final observations и reward последнего реального transition не теряются.
- После done следующий step без reset отклоняется.
- PV пропал/неуникален или нарушен invariant — error/invalid episode, не успешный terminal и не тихий reward=0. В данном closed-ring дизайне такой исход не штатен.

### 5.3. Пустая AV-окрестность

При |A_t|=0 **оба логических агента остаются живы**. AV observation имеет padding, action mask stay-only; PV продолжает управляться; reward существует; AV actor/entropy/KL contributions=0, value loss остаётся определённым. AV отсутствие не должно разрывать GAE, менять batch step accounting или завершать эпизод.

## 6. Наблюдения

### 6.1. AV: неизменённый контракт статьи

| Поле | Форма | dtype | Правило |
|---|---|---|---|
| node_features | [164,48] | float32 | bounded engineered features |
| node_mask | [164] | int8 | 1 у активных slots |
| neighbor_index | [164,6] | int32 | 0 для отсутствующей связи, игнорируется mask |
| neighbor_mask | [164,6] | int8 | binary |
| edge_features | [164,6,3] | float32 | relative position/lane/speed |
| action_mask | [164,3] | int8 | left,stay,right |
| global_features | [4] | float32 | PV speed, lane, density, active fraction |

Порядок 48 полей и математические нормировки сохраняются из `PV_GRAPH_NODE_FEATURES` / `paper/main.tex:273–346`. Нельзя добавлять PV cooldown, PV planned action, policy weights или role embedding в существующие AV-входы основного режима.

Slot assignment: signed periodic displacement, затем lane, затем stable ID. ID доступен только environment для routing. Нельзя применить действие к mapping от X_(t+1).

Рёбра: ближайшие **активные AV** спереди/сзади в left/same/right lane; sensor cutoff к рёбрам не применяется. Local traffic descriptors могут видеть HDV/PV/AV до 60 клеток от соответствующего AV; граф остаётся ограничен окном [-10,+30]. `gap_norm` в observations — расстояние между head cells, а не empty-cell gap физики.

После смены полосы PV обновить PV-relative lane/global lane/node contexts для всех AV. Полоса PV не сужает окно до одной полосы. Продольный PV step сдвигает moving window; собственно смена его полосы не меняет x-window.

### 6.2. PV: локальная схема L и общие ego features обоих вариантов

Добавить независимый `Dict`:

| Поле | Форма | dtype |
|---|---|---|
| ego_features | [48] | float32 |
| global_features | [4] | float32 |
| action_mask | [3] | int8 |

`ego_features` использует те же определения 48 признаков, но ego — настоящий PV; никаких временных изменений TrafficState. Это переиспользование **семантики признаков**, не существующий готовый публичный builder для PV.

| Индекс | Определение для ego=PV |
|---|---|
| 0 | pv_relative_position=0 |
| 1 | lane_PV / max(M−1,1) |
| 2 | pv_relative_lane=0 |
| 3 | speed_PV / max(vmax_hdv,vmax_priority,1)=v_PV/6 |
| 4 | pv_relative_speed=0 |
| 5 | последний фактически выполненный lane_delta PV ∈ {−1,0,+1} |
| 6 | PV cooldown / max(C_PV,1) |
| 7–17 | local lane block для lane_PV−1 |
| 18–28 | block для lane_PV |
| 29–39 | block для lane_PV+1 |
| 40–43 | ближайший HDV впереди по всем полосам: presence, distance/60, relative_lane/3, relative_speed/6 |
| 44–47 | аналогично HDV позади |

Каждый local lane block содержит ровно:

`lane_exists, front_present, front_distance/60, (v_front−v_ego)/6, front_is_hdv, front_is_pv, back_present, back_distance/60, (v_back−v_ego)/6, back_is_hdv, back_is_pv`.

- Front/back — directed periodic distances от ego, ego исключается из кандидатов.
- Missing lane: весь block=0.
- Existing lane без соседа до 60 включительно: lane_exists=1, соответствующие neighbour fields=0.
- При одном PV neighbour_is_pv=0 для его собственного ego-наблюдения.
- AV определяется presence=1, is_hdv=0, is_pv=0.
- Directed distance=0 на другой полосе допустим; tie-break как в существующем builder.
- Константные нули сохраняются ради единой семантики, не объявляются полезной информацией.
- Global vector тот же: `[v_PV/6, lane_PV/3, N/(ML), |A_t|/164]`.
- В варианте L PV actor не получает AV graph. Ни L, ни G не получают текущую/будущую команду AV-партнёра, его logits, weights или value estimates.

Это новый информационный контракт, не «полностью идентичное observation статье» — у PV в статье вообще не было policy. Ego schema48 и global4 одинаковы в обоих вариантах PV; история/RNN не добавляется.

### 6.3. Вариант G: тот же локальный PV + графовое окружение

Для `pv_encoder=graph_mlp` observation имеет структуру:

```text
Dict({
  ego_features: float32[48],
  av_graph: Dict(полный неизменённый AV graph schema из §6.1),
  action_mask: int8[3]  # именно PV left/stay/right
})
```

`global_features[4]` доступны внутри `av_graph`; вторую отдельную копию в G observation не создавать. AV и PV получают одинаковые **сырые graph tensors** одного X_t; эти данные можно построить один раз. Encoder PV заново вычисляет своё представление из raw tensors — чужие обучаемые embeddings в environment не передаются. Graph arrays не должны мутироваться inference code.

- `av_graph` по-прежнему только AV nodes, максимум164. PV не добавляется в него ни узлом, ни AV action slot.
- PV-encoder использует node/global features, node/edge masks, neighbor indices и edge descriptors; вложенную AV `action_mask[164,3]` не использует для PV logits или pooling. Она сохранена только для совместимости raw graph schema.
- PV `action_mask[3]` всегда верхнего уровня, соответствует единственному PV action.
- При нуле AV masked graph pool=0; global и локальные PV features остаются доступны. Нулевой pool не означает одинаковые logits L и G, но G обязан реагировать на локальную обстановку.
- В PV-only G режиме (`P_G`) наблюдать граф tagged AV можно даже без управления ими: это sensing, не скрытая AV-cooperation. Они продолжают native движение.
- Узлы/рёбра/нормировки G совпадают с AV graph статьи; роль PV/политические полномочия не меняются ради features.

Таким образом сравнение L↔G проверяет полезность **добавленного графового контекста с encoder**, а не только message passing при одинаковом информационном наборе. Различия capacity и compute раскрываются в отчёте.

### 6.4. Маски

Для PV и активного AV:

- stay всегда valid;
- left/right valid только при существующей adjacent lane, отсутствии body occupancy target, strict rear-safety condition и нулевом cooldown;
- проверка из полного текущего состояния, а не только sensor descriptors;
- native incentive не обязателен для override;
- будущая скорость другого автомобиля не предсказывается;
- padding AV имеет [0,1,0].

Actor получает masked logits до sampling/argmax; invalid logits = −1e9 как в статье. Маска не является learned feature безопасности и не заменяет defensive validation в simulator.

## 7. Действия, конфликты и точный порядок шага

### 7.1. Encoding

Одинаково для обеих новых policy:

| Network index | Lane delta | Значение |
|---|---|---|
| 0 | −1 | left |
| 1 | 0 | forced stay |
| 2 | +1 | right |

Не использовать без явного адаптера generic `SnfsTrafficEnv/MultiAgentEnv`: там индексы stay,left,right. Проверить и scalar PV, и vector AV.

### 7.2. Один transition

```text
X_t := current simulator state
O_AV, mapping_t, masks_AV := build from X_t
O_PV, mask_PV := build from X_t
receive a_AV, a_PV selected from these observations
validate entire action dictionary before mutating anything
U_AV := map each active slot using mapping_t; include explicit stay
U_PV := {physical_pv_id: decoded_action}; include explicit stay
apply defensive cooldown guard; record raw and guarded requests
assert keys(U_AV) and keys(U_PV) disjoint
U := union(U_AV, U_PV)
X_next := simulator.step_lateral_overrides(U)  # exactly once
split override result by physical role for event telemetry
update per-vehicle cooldown once
compute reward and physical metrics from X_next
increment physical step counter once
build both next observations from X_next
return one joint transition
```

Если в контрольном режиме PV=native, его ID **отсутствует** в U_PV. Если AV=native, AV IDs отсутствуют в U_AV. Если оба native, использовать штатный native step для точного B1 compatibility; отсутствие override и forced stay не взаимозаменяемы.

### 7.3. Conflict semantics

Сохранить реализацию `control.py:701–877`:

1. Все explicit overrides сортируются по физическим ID.
2. Любой explicit ID исключается из native lateral generation — даже stay, unsafe и later conflict rejection.
3. Individually valid explicit proposals группируются по target `(lane,x)`.
4. При нескольких explicit requests в одну target cell отклоняются **все**, без победителя PV/AV.
5. Accepted explicit cells резервируются перед native resolution.
6. Native candidates на уже зарезервированные target keys отфильтровываются существующим kernel path; затем native conflicts разрешаются тем же RNG.
7. Принятые native и explicit lateral moves применяются одновременно; затем один longitudinal update.

Не давать PV новое преимущество в **explicit-vs-explicit** arbitration: это была бы самостоятельная смена safety/dynamics. Уже существующий приоритет explicit над native неизбежно начинает распространяться на PV после подключения его policy; это часть эффекта интерфейса и причина обязательного PV-heuristic контроля.

Запрещены sequential «сначала расчистить AV, потом построить mask PV», lane swaps в клетки, занятые в X_t, teleport, дополнительное разрешение intra-step follow-through. Они меняют temporal resolution статьи.

### 7.4. Cooldown

**PROPOSED:** C_PV=C_AV=1. Для non-stay applied в t установить c(t+1)=1; следующее решение stay-only; после него c=0. Rejected или stay не запускают новый cooldown. Хранить c по physical ID; уменьшать и у AV вне окна, чтобы re-entry не сбрасывал память. PV cooldown сбрасывается только на reset, не при смене полосы.

Для native PV baseline не накладывать artificial policy cooldown: он остаётся native. Это отдельно отражается в контролях.

## 8. Reward и учёт интервенций

### 8.1. Основное предложение: общая cooperative reward

Пусть для g∈{PV,AV} N_app^g — число реально применённых non-stay overrides, N_rej^g — сумма individually blocked, explicit-conflict и defensively cooldown-suppressed non-stay запросов за transition.

**FIXED, пользователь согласовал сохранение reward статьи:**

`r_t = v_PV(t+1)/6 − 0.001*N_app^AV − 0.002*N_rej^AV`.

`rewards['pv'] = rewards['av_coordinator'] = r_t`.

Сохраняются не только коэффициенты, но и область подсчёта событий исходной статьи: только управляемые AV. PV actions учитываются в диагностике, однако прямого штрафа за перестроение PV в основном reward нет. Это осознанное ограничение: PV может маневрировать часто, поэтому role-split lane-change и oscillation metrics обязательны. При пустом AV graph reward обоим равен v_PV(t+1)/6.

- Не делить event costs на число AV.
- Не умножать speed term на число агентов.
- Не включать background flow, собственную скорость AV, stopped penalty, distance shaping или curriculum coefficients.
- Обе policies могут получать одинаковую reward, сохраняя разные policies и разные value/advantage оценки.
- Сумма rewards двух agents в RLlib episode return может стать 2× физического team return. Для отчёта хранить `team_return=sum_t r_t` один раз, плюс per-policy returns. Не сравнивать напрямую generic summed MARL return со старым single-agent return.

Альтернативы «AV платят только за свои события, PV только за свои» и «оба получают общий штраф за AV+PV события» меняют исходный reward contract. Симметричная версия `r_joint_cost=v_PV/6−0.001*(N_app^AV+N_app^PV)−0.002*(N_rej^AV+N_rej^PV)` разумна, но вводится только по явному решению D03 как отдельная ablation с retraining. Не выбирать reward после просмотра test.

### 8.2. Event accounting

Сохранять по ролям и суммарно:

- raw_left/right/stay;
- requested_nonstay_after_cooldown;
- applied_left/right;
- blocked_out_of_bounds / target_occupied / unsafe;
- explicit_conflict;
- cooldown_suppressed;
- native lane changes отдельно от policy applied;
- active AV action opportunities=sum_t |A_t|, PV opportunities=T.

Для роли g:

`raw_nonstay = cooldown_suppressed + issued_nonstay`;

`issued_nonstay = applied + blocked + explicit_conflict`;

`reward_rejected = blocked + explicit_conflict + cooldown_suppressed`.

Stay и padding не являются rejected. В текущем `PriorityMetricsAccumulator.observe(result=...)` есть generic подсчёт `len(result)-applied`, который без дополнительной фильтрации включает stay. Поэтому нельзя просто передать общий PV+AV result и довериться старому полю `rule_rejected`. Физические метрики переиспользуются, новую role telemetry вычислять отдельно и проверять инварианты сумм.

В compliant policy cooldown_suppressed=0; это подтверждает работу маски, не отсутствие cooldown. Individually invalid sampled action тоже должен быть нулём; конфликт PV↔AV после valid masks допустим.

## 9. Модели и оптимизация

### 9.1. AV policy

Сохранить `PvGraphTorchModel` с hidden_dim=128, message_layers=3:

- node encoder 48→128→128;
- global encoder 4→128→128;
- три message/update layers, masked neighbour mean, residual+LayerNorm+ReLU;
- shared per-node actor 256→128→3;
- masked node mean + global → value head 256→128→1;
- 355460 обучаемых параметров.

Не перепутать с defaults h64/l2 и docstring «two layers». Runtime resolved config имеет приоритет.

### 9.2. PV policies: два заранее заданных варианта

#### L — `local_mlp`

Concatenate `[ego_features,global_features]` в52-vector; trunk `Linear(52,128),ReLU,Linear(128,128),ReLU`; actor `Linear(128,3)` с PV mask; critic `Linear(128,1)`. Все Linear с bias. PV23812 параметров; AV+PV=379272. Это прежний минимальный вариант.

#### G — `graph_mlp`

**PROPOSED точная архитектура:** независимый graph backbone PV повторяет AV backbone из `graph_ppo.py`:

- node encoder `48→128→128`, ReLU только между Linear;
- global encoder `4→128→128`, ReLU только между Linear;
- initial nodes `ReLU(node_embedding + global_embedding) * node_mask`;
- те же3 message-passing layers (edge_dim3, masked mean, residual/LayerNorm/ReLU), h128;
- `graph_pool = sum(node_embeddings*mask)/max(1,sum(mask))`, форма[B,128].

Дополнительно PV ego encoder `48→128→128` с ReLU после каждого Linear; он обрабатывает прежние локальные PV features, включая PV cooldown. Затем:

`z_PV = concat(ego_embedding128, graph_pool128, global_embedding128)` — размер384;

`actor = Linear(384,128),ReLU,Linear(128,3)`, затем **PV** mask;

`critic = Linear(384,128),ReLU,Linear(128,1)`.

Это именно отдельные MLP-головы PV поверх графового представления, а не per-node AV actor, которому случайно дали PV ID. Node actor/value головы AV в PV backbone не копировать: они не используются.

Расчёт parameter counts при указанной схеме:

| Компонент | Параметров |
|---|---:|
| PV graph backbone (node+global+3MP, без старых heads) | 289152 |
| PV ego encoder | 22784 |
| PV actor head | 49667 |
| PV value head | 49409 |
| PV total G | 411012 |
| AV+PV total G | 766472 |

Числа — расчёт спецификации; проверить фактический model count перед pilot. Оба варианта обучаются с нуля, одинаковый PPO schedule и20M env steps/run. Это env-step-matched, **не capacity/compute-matched** сравнение. Победа G обосновывает полезность данного graph-context варианта, но сама по себе не доказывает преимущество message passing над равномощной MLP. Для более сильного вывода понадобится отдельная заранее утверждённая capacity/information-matched ablation.

#### Параметры и gradient ownership

AV и PV имеют общий raw graph/feature формат и могут переиспользовать Python implementation класса encoder, но **не learned tensors**. Это два самостоятельных экземпляра сети: отсутствуют cross-policy gradient, weight aliasing, joint critic и чужие optimizer moments. Actor и critic внутри одной PV-policy разделяют её encoder. В G PV encoder обучается только PV loss, AV encoder — только AV loss; одинаковая архитектура не означает одинаковые numerical embeddings после обучения.

Shared AV encoder с двумя MLP-головами был бы возможным **третьим** вариантом: требуется решить gradient ownership, общий optimizer, checkpoint ownership и fairness. Он не нужен для двух согласованных вариантов и не вводится молча.

### 9.3. Минимальное переиспользование loss для PV

Существующий `factorized_ppo_loss` требует `model.max_nodes`, `model.last_node_mask`, logits и scalar values. **PROPOSED:** PV-model адаптирует этот контракт:

- `max_nodes=1`;
- `last_node_mask=ones([B,1])`;
- logits [B,3], внутри loss reshape [B,1,3];
- Discrete actions [B] reshape [B,1];
- value_function [B];
- `restore_original_dimensions` для flattened RLlib Dict, включая nested `av_graph` варианта G.

В G не путать размер входного графа и число действий: `graph_max_nodes=164` и внутренний graph_node_mask служат encoder/pooling, а loss-facing `max_nodes=1`, `last_node_mask=ones([B,1])` относятся к **одному PV**. Даже при пустом входном AV graph PV actor/entropy/KL loss не зануляется по AV mask. В каждой обучающей job выбирается только один PV encoder L или G; architecture ID и observation schema входят в checkpoint manifest, restore L↔G запрещён.

Тогда node-factorized loss при N=1 становится обычным categorical clipped surrogate, сохраняя **точный value-clipping вариант статьи**. Новый дублирующий PPO loss не нужен. Это должно пройти numeric и RLlib integration tests; статическая совместимость не равна runtime проверке.

Две явные `PolicySpec` используют отдельные экземпляры `FactorizedPPOTorchPolicy` с разными model classes/spaces. «Разные политики» означает независимые параметры и optimizer state, а не обязательно разные Python loss classes. Не рассчитывать на `FactorizedGraphPPO.get_default_policy_class` для автоматического выбора модели по типу ТС.

### 9.4. Loss contract

AV actor: clipped ratio отдельно на каждом active node, masked mean по nodes каждого transition, затем mean по transitions. Общий AV advantage для node actions; не произведение joint ratios. PV actor: тот же functional при N=1, свой advantage.

Для каждой policy отдельно:

`delta_t = r_t + gamma*b_t*V(O_(t+1)) − V(O_t)`;

`A_t = delta_t + gamma*lambda*continuation_t*A_(t+1)`.

Истинный terminal обнуляет bootstrap. Time-limit truncation требует bootstrap из final observation; recursion не должна переноситься на reset следующего эпизода. Rollout-fragment cut тоже требует корректного bootstrap. Проверять вручную на короткой траектории; версии RLlib могут отличаться.

- Store old logits и old values от той самой behaviour policy и O_t.
- Advantage normalization раздельно внутри policy batch.
- Same transition indices и joint rollout; никакого отбора AV-only физических ходов.
- Padding игнорируется actor/entropy/KL; пустой AV graph не удаляется из value batch.
- ratio log-difference clamp [-20,20], clipping epsilon=0.2.
- Value loss: max(unclipped squared error, old-value-relative clipped squared error), vf_clip=10.
- entropy/KL маскирование и редукция как в `graph_ppo.py`.
- Не называть метод стандартным joint-action PPO или автоматически MAPPO/CTDE. Это две PPO-style policies с исходным AV nodewise surrogate.

### 9.5. Training loop и budgets

**FIXED сравнимый schedule:** gamma=.995, GAE lambda=.95, lr=.0002, entropy=.01, vf_loss_coeff=1, vf_clip=10, KL initial=.20, target=.01, six epochs, minibatch1024.

32 CPU rollout workers, 1 environment на worker, fragment256, **16384 environment steps** на iteration, GPU1 — reference execution profile, не обещание доступного железа. Все sampler policies синхронизируются с одной joint checkpoint version перед следующим rollout batch.

Обязательно `count_steps_by='env_steps'`. При двух постоянных agents 16384 env steps дают примерно 32768 logical agent transitions, по 16384 на policy. Число AV node actions дополнительно переменно и не является sampled env steps.

Целевой budget 20M **совместных физических transitions на training run**, не 20M на каждого агента и не 10M+10M. При штатном batch16384 пересечение порога: iteration1221, 20004864 шагов. Логировать actual counts; при иной сборке batch не выводить steps из iteration арифметически.

Основное joint обучение — from scratch, обе policies trainable с iteration1. Frozen-партнёр и fine-tuning — отдельные named controls. На всех основных configs отключить test evaluation и adaptive hyperparameter search.

Полный расширенный план: A (только AV), P_L/P_G (только PV), J_L/J_G (joint), по3seeds×20M =300M env transitions. A не дублируется ради второго PV encoder. AV initialization согласуется по seed между A/J_L/J_G, но после старта это независимые runs/weights. Вместо выбора architecture по test публикуются оба варианта, с отдельными validation-selected checkpoints. Pilot-first принят; фактический execution budget замораживается после измерения ресурсов обоих вариантов.

## 10. Checkpoint, restore и failure handling

### 10.1. Единица результата — согласованная пара

Joint checkpoint должен содержать:

- обе policies и value heads из одного training snapshot;
- optimizer states, per-policy KL/entropy scheduler states;
- sampled env/agent steps, iteration, policy update counters;
- resolved simulator/observation/reward/action/model configs и schema versions;
- base/training seeds, software versions, git commit/dirty state;
- режим initialization и, если есть, parent checkpoint hashes;
- manifest пары `joint_checkpoint_id`, AV/PV hashes, policy IDs.

Нельзя выбирать лучший AV checkpoint отдельно от лучшего PV checkpoint и объявлять их joint training результатом. Такая recombination — cross-play intervention, только отдельная диагностика.

### 10.2. Save protocol

Новое имя на каждый snapshot; запись во временную директорию в research output tree, manifest и completeness checks, затем atomic publish/rename на той же filesystem. Не перезаписывать существующие results. `best` определяется только validation; interrupted не маскируется под final.

При прерывании сохранять согласованную пару в iteration boundary, если алгоритм способен это сделать безопасно. Ошибка одного policy-save делает snapshot incomplete. Частичный snapshot не должен проходить selector.

### 10.3. Restore smoke

Проверять обе модели:

1. explicit policy IDs и schema;
2. strict key/shape checks для weights;
3. deterministic masked actions;
4. logits и values с declared tolerance (например CPU float32 atol=1e−6, rtol=1e−5, зафиксировать фактически проверенный режим);
5. joint short rollout RLlib vs extracted CPU inference;
6. normal/empty/dense AV graph, lane boundary, cooldown, PV role;
7. после resume следующий train iteration действительно обновляет обе trainable policies.

Автопоиск `default_policy` или восстановление только первой policy запрещены.

### 10.4. Два уровня воспроизводимости

**Обязательно:** deterministic evaluation при фиксированных checkpoint, конфигурации, состоянии, seeds и software/backend; сохраняются RNG history contracts.

**Не обещать автоматически:** bitwise-identical distributed training continuation после RLlib restore. Algorithm checkpoint может не включать simulator RNG, np_random, warm-up cache, condition order, in-flight episodes и worker scheduling. Manifest должен различать `algorithm_state_resume` и `exact_rollout_resume`.

Для обычного interrupted продолжения достаточно явно документированного algorithm-state restore с сохранёнными weights/optimizers/counters и новым sampler segment. Для утверждения exact continuation нужен дополнительный полный env/worker state serialization и тест непрерывного run против split+restore. Не строить этот слой без потребности, но и не выдавать первый режим за второй.

OOM, missing checkpoint, некорректные actions, broken pairing и backend fallback не исправлять скрытым изменением размеров, seeds или policy. Сохранить reason/status, остановить соответствующий этап и оформить deviation. Retry не должен перетирать ранее завершённую репликацию.

## 11. Обязательная телеметрия

Training JSONL по iteration:

- iteration, sampled_env_steps, sampled_agent_steps, active_AV_decisions, elapsed;
- pv_policy и av_policy: actor/value/total loss, entropy, KL и текущий coefficient, explained variance, update count;
- team_return и оба per-policy return с пояснённой суммой;
- PV speed overall и по 12 conditions, active AV mean/min/max;
- role-split reward components и event counts;
- empty-graph fraction, mask-violation fraction, PV↔AV conflicts;
- sampling/learning throughput, host memory/GPU peak memory при доступности;
- backend requested/actual/override-path fallback;
- checkpoint path/status/hash, run segment ID.

Evaluation: полный run-level schema, paired keys, метрики и protocol gates описаны в [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Generic training return не является primary research outcome.

## 12. Критерий завершения

ТЗ считается реализованным только после утверждения решений, прохождения acceptance tests, воспроизводимого joint training, selection согласованных пар, held-out paired evaluation всех обязательных controls и отчёта с честными ограничениями. Требование положительного эффекта к технической приёмке не предъявляется.

Документационный запрос, на котором утверждена редакция0.2, завершён. **Последующим поручением пользователь запустил выполнение**, потребовал все вычисления в devcontainer и подтвердил доступность2×RTX3090. Текущее состояние кода/проверок — в [EXECUTION_LOG.md](EXECUTION_LOG.md); наличие реализации не означает завершённый training/evaluation. Приоритет — добавлять новые компоненты, допускается небольшое дублирование, старые source paths менять только как крайний вариант с regression-проверкой воспроизведения статьи. Согласован порядок correctness→pilot→измерение ресурсов→main; конкретный расширенный бюджет двух архитектур и CPU/GPU execution profile фиксируются после pilot.
