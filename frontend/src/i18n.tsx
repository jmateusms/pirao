/**
 * Portuguese and English, as in farofa gui and faultree gui.
 *
 * One dictionary, both languages side by side, so a missing translation is a
 * type error rather than an English phrase in the middle of a Portuguese
 * screen.  `{name}` placeholders are filled by `t(key, { name })`.
 *
 * Numbers follow the language too: Portuguese writes 0,95 and 3,5e-4.  The
 * formatters read the language from module state that the provider sets
 * before rendering, so plain helpers outside components (fmt, sig) need no
 * extra argument.
 */

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

export type Lang = 'pt' | 'en'

const STRINGS = {
  // ---- shell
  'app.sub': {
    en: 'Probabilistic Inference for Reliability Analysis from Observed data',
    pt: 'inferência bayesiana de confiabilidade a partir de dados observados',
  },
  'app.examples': { en: 'Examples…', pt: 'Exemplos…' },
  'app.examplesAria': { en: 'Load a ready-made example', pt: 'Carregar um exemplo pronto' },
  'app.run': { en: 'Run MCMC', pt: 'Rodar MCMC' },
  'app.lang': { en: 'Language', pt: 'Idioma' },
  'app.sideAria': { en: 'Model, data and sampler', pt: 'Modelo, dados e amostrador' },
  'app.resizeSide': { en: 'Resize the side panel', pt: 'Redimensionar o painel lateral' },
  'app.hasProblems': { en: 'has problems', pt: 'tem problemas' },
  'app.noStan': { en: 'Stan is not installed.', pt: 'O Stan não está instalado.' },
  'app.unreachable': { en: 'The backend is unreachable.', pt: 'O servidor não responde.' },
  'app.startWith': { en: 'Start it with', pt: 'Inicie com' },
  'app.loading': { en: 'Loading…', pt: 'Carregando…' },
  'app.exampleFailed': {
    en: 'Could not load the example {id}: {error}',
    pt: 'Não foi possível carregar o exemplo {id}: {error}',
  },
  'app.couldNotStart': { en: 'Could not start', pt: 'Não foi possível iniciar' },
  'tab.Model': { en: 'Model', pt: 'Modelo' },
  'tab.Data': { en: 'Data', pt: 'Dados' },
  'tab.Sampler': { en: 'Sampler', pt: 'Amostrador' },
  'ready.title': { en: 'Ready when you are', pt: 'Tudo pronto' },
  'ready.body': {
    en: 'Set up the model, the data and the sampler on the left, or pick a ready-made analysis from Examples. The results appear here: reliability over time first, then what the data changed in each parameter.',
    pt: 'Monte o modelo, os dados e o amostrador à esquerda, ou escolha uma análise pronta em Exemplos. Os resultados aparecem aqui: primeiro a confiabilidade ao longo do tempo, depois o que os dados mudaram em cada parâmetro.',
  },
  'ready.fixFirst': {
    en: 'Fix the problems marked on the {tab} tab first.',
    pt: 'Corrija antes os problemas marcados na aba {tab}.',
  },

  'plot.failed': {
    en: 'The plotting library failed to load.',
    pt: 'A biblioteca de gráficos não carregou.',
  },

  // ---- example note
  'example.public': { en: 'public data', pt: 'dados públicos' },
  'example.illustrative': { en: 'illustrative data', pt: 'dados ilustrativos' },
  'example.close': { en: 'close', pt: 'fechar' },
  'example.source': { en: 'Source:', pt: 'Fonte:' },

  // ---- model
  'model.likelihood': { en: 'Likelihood', pt: 'Verossimilhança' },
  'model.likelihoodIntro': {
    en: 'What kind of observation each row of your data represents.',
    pt: 'Que tipo de observação cada linha dos seus dados representa.',
  },
  'model.missionTimes': { en: 'Mission time(s)', pt: 'Tempo(s) de missão' },
  'model.missionTimesPlaceholder': { en: 'e.g. 100, 500, 1000', pt: 'ex.: 100; 500; 1000' },
  'model.missionTimesHelp': {
    en: 'Reliability is reported at each of these; separate several with commas.',
    pt: 'A confiabilidade é informada em cada um; separe vários com ponto e vírgula.',
  },
  'model.missionDemands': { en: 'Mission demands', pt: 'Demandas da missão' },
  'model.missionDemandsHelp': {
    en: 'How many demands the unit must survive.',
    pt: 'Quantas demandas a unidade precisa suportar.',
  },
  'model.timeUnit': { en: 'Time unit', pt: 'Unidade de tempo' },
  'model.timeUnitHelp': {
    en: 'Printed on every plot and export. It matters: weighting by relevance is not invariant to the unit.',
    pt: 'Aparece em todos os gráficos e exportações. Ela importa: a ponderação por relevância depende da unidade.',
  },
  'model.curveTo': { en: 'Curve runs to', pt: 'Curva vai até' },
  'model.auto': { en: 'auto', pt: 'auto' },
  'model.curveToHelp': {
    en: 'Where the reliability curve stops, in {unit}. Leave empty for 1.3 × the mission, which shows where the estimate is heading without implying the extrapolation is data.',
    pt: 'Onde a curva de confiabilidade termina, em {unit}. Deixe vazio para 1,3 × a missão, o que mostra para onde a estimativa vai sem sugerir que a extrapolação é dado.',
  },
  'model.demands': { en: 'demands', pt: 'demandas' },
  'model.priorOnly': {
    en: 'Ignore the data (sample the prior only)',
    pt: 'Ignorar os dados (amostrar só a priori)',
  },
  'model.priorOnlyHelp': {
    en: 'Fits the prior alone, so you can see what it implies before the data has any say.',
    pt: 'Ajusta só a priori, para ver o que ela implica antes de os dados opinarem.',
  },
  'model.cannotFit': {
    en: 'This model cannot be fitted yet.',
    pt: 'Este modelo ainda não pode ser ajustado.',
  },
  'model.showSource': { en: 'Show the generated Stan program', pt: 'Mostrar o programa Stan gerado' },
  'model.hideSource': { en: 'Hide the generated Stan program', pt: 'Esconder o programa Stan gerado' },
  'model.prior': { en: 'Prior', pt: 'Priori' },
  'model.lower': { en: 'Lower bound', pt: 'Limite inferior' },
  'model.upper': { en: 'Upper bound', pt: 'Limite superior' },
  'model.none': { en: 'none', pt: 'nenhum' },
  'model.presets': { en: 'Presets', pt: 'Atalhos' },
  'model.median': { en: 'median', pt: 'mediana' },
  'model.interval90': { en: '90% interval', pt: 'intervalo de 90%' },
  'model.noPreview': { en: 'No preview available', pt: 'Sem pré-visualização' },
  'model.inForce': { en: 'In force:', pt: 'Em vigor:' },
  'model.to': { en: 'to', pt: 'a' },
  'model.keeps': { en: 'keeps {pct}% of the prior', pt: 'mantém {pct}% da priori' },
  'model.massWarning': {
    en: 'Your bounds keep only {pct}% of the {family} prior you entered. The curve shown is the restricted prior, which is what the model will use.',
    pt: 'Seus limites mantêm só {pct}% da priori {family} que você informou. A curva mostrada é a priori restrita, que é a que o modelo vai usar.',
  },

  // ---- data
  'data.title': { en: 'Data', pt: 'Dados' },
  'data.templateCsv': { en: 'Template (CSV)', pt: 'Modelo (CSV)' },
  'data.templateXlsx': { en: 'Template (Excel)', pt: 'Modelo (Excel)' },
  'data.upload': { en: 'Upload…', pt: 'Abrir arquivo…' },
  'data.intro': {
    en: 'One row per unit. {failure1} means the unit failed; {zero} means it survived and the row is censored. {relevance} is optional and defaults to 1. You can paste a block straight from Excel.',
    pt: 'Uma linha por unidade. {failure1} significa que a unidade falhou; {zero} significa que ela sobreviveu e a linha é censurada. {relevance} é opcional e vale 1 por padrão. Dá para colar um bloco direto do Excel.',
  },
  'data.newRow': { en: 'New row…', pt: 'Nova linha…' },
  'data.resize': { en: 'Resize the table', pt: 'Redimensionar a tabela' },
  'data.row1': { en: '1 row', pt: '1 linha' },
  'data.rows': { en: '{n} rows', pt: '{n} linhas' },
  'data.addRow': { en: 'Add row', pt: 'Adicionar linha' },
  'data.removeLast': { en: 'Remove last', pt: 'Remover a última' },
  'data.clear': { en: 'Clear', pt: 'Limpar' },
  'data.problems': { en: 'Problems to fix', pt: 'Problemas a corrigir' },
  'data.row': { en: 'Row {n}', pt: 'Linha {n}' },

  // ---- sampler
  'sampler.title': { en: 'Sampling', pt: 'Amostragem' },
  'sampler.intro': {
    en: 'The defaults suit these models. Change them if the run reports a problem — each setting below says which problem it addresses.',
    pt: 'Os valores padrão servem para estes modelos. Mude-os se a rodada apontar um problema; cada ajuste abaixo diz qual problema resolve.',
  },
  'sampler.chains': { en: 'Chains', pt: 'Cadeias' },
  'sampler.iter_warmup': { en: 'Warmup draws', pt: 'Amostras de aquecimento' },
  'sampler.iter_sampling': { en: 'Kept draws', pt: 'Amostras guardadas' },
  'sampler.adapt_delta': { en: 'Target acceptance', pt: 'Aceitação alvo' },
  'sampler.max_treedepth': { en: 'Max tree depth', pt: 'Profundidade máxima da árvore' },
  'sampler.seed': { en: 'Random seed', pt: 'Semente' },
  'sampler.random': { en: 'random', pt: 'aleatória' },
  'sampler.total': {
    en: '{chains} chains × {draws} draws = {total} draws in total. Sampling these models takes a moment; the wait you will notice is the one-off compilation of a model structure you have not used before.',
    pt: '{chains} cadeias × {draws} amostras = {total} amostras no total. Amostrar estes modelos leva um instante; a espera que você vai notar é a compilação, feita uma única vez, de uma estrutura de modelo ainda não usada.',
  },

  // ---- run
  'stage.resolve': { en: 'Checking the model', pt: 'Conferindo o modelo' },
  'stage.render': { en: 'Assembling the Stan program', pt: 'Montando o programa Stan' },
  'stage.compile': { en: 'Compiling', pt: 'Compilando' },
  'stage.sample': { en: 'Sampling', pt: 'Amostrando' },
  'stage.summarize': { en: 'Summarising', pt: 'Resumindo' },
  'stage.done': { en: 'Finished', pt: 'Concluído' },
  'run.title': { en: 'Run', pt: 'Rodada' },
  'run.ready': {
    en: 'Everything checks out. Start when you are ready.',
    pt: 'Tudo certo. Comece quando quiser.',
  },
  'run.fixFirst': {
    en: 'Fix the problems flagged on the Model and Data tabs first.',
    pt: 'Corrija antes os problemas apontados nas abas Modelo e Dados.',
  },
  'run.cancel': { en: 'Cancel', pt: 'Cancelar' },
  'run.didNotFinish': { en: 'The run did not finish.', pt: 'A rodada não terminou.' },
  'run.row': { en: 'Row {n}', pt: 'Linha {n}' },
  'run.cancelled': { en: 'This run was cancelled.', pt: 'Esta rodada foi cancelada.' },
  'run.again': { en: 'Run again', pt: 'Rodar de novo' },

  // ---- results: header and status
  'res.title': { en: 'Results', pt: 'Resultados' },
  'res.meta': { en: 'Run {id} · seed {seed}', pt: 'Rodada {id} · semente {seed}' },
  'res.draws': { en: 'Draws (CSV)', pt: 'Amostras (CSV)' },
  'res.bundle': { en: 'Full analysis (.zip)', pt: 'Análise completa (.zip)' },
  'res.bundleHint': {
    en: 'Specification, data, seed, the Stan program and the toolchain versions: everything needed to reproduce the run',
    pt: 'Especificação, dados, semente, o programa Stan e as versões do compilador: tudo o que é preciso para reproduzir a rodada',
  },
  'res.priorOnly': {
    en: 'Prior only: the data were ignored, so these numbers show what the prior alone implies.',
    pt: 'Só a priori: os dados foram ignorados, então estes números mostram o que a priori sozinha implica.',
  },
  'res.weighted': {
    en: 'Relevance weighting: {rows} rows count as {n} observations. This is a fractional (pseudo-)posterior, so the intervals are deliberately wider than a standard Bayesian analysis would give.',
    pt: 'Ponderação por relevância: {rows} linhas valem {n} observações. É uma (pseudo-)posteriori fracionária, então os intervalos são de propósito mais largos do que numa análise bayesiana padrão.',
  },
  'res.stanTitle': { en: 'The Stan program that was fitted', pt: 'O programa Stan que foi ajustado' },
  'elapsed.resolve': { en: 'resolve', pt: 'verificação' },
  'elapsed.render': { en: 'render', pt: 'montagem' },
  'elapsed.compile': { en: 'compile', pt: 'compilação' },
  'elapsed.sample': { en: 'sample', pt: 'amostragem' },
  'elapsed.summarize': { en: 'summarise', pt: 'resumo' },
  'view.Overview': { en: 'Overview', pt: 'Visão geral' },
  'view.Table': { en: 'Table', pt: 'Tabela' },
  'view.Diagnostics': { en: 'Diagnostics', pt: 'Diagnóstico' },
  'view.Stan program': { en: 'Stan program', pt: 'Programa Stan' },
  'facts.failure1': { en: '1 failure', pt: '1 falha' },
  'facts.failures': { en: '{n} failures', pt: '{n} falhas' },
  'facts.inTrials': { en: '{f} in {n} trials', pt: '{f} em {n} tentativas' },
  'facts.censored': { en: '{n} censored', pt: '{n} censurados' },
  'facts.onTest': { en: '{x} {unit} on test', pt: '{x} {unit} em teste' },
  'facts.demands': { en: '{n} demands', pt: '{n} demandas' },
  'status.check': {
    en: 'Check the sampler before trusting these numbers.',
    pt: 'Confira o amostrador antes de confiar nestes números.',
  },
  'status.numbers': { en: 'max R-hat {r} · min ESS {e}', pt: 'R-hat máx. {r} · ESS mín. {e}' },
  'status.ok': { en: 'The sampler converged.', pt: 'O amostrador convergiu.' },
  'status.note1': { en: '1 note', pt: '1 observação' },
  'status.notes': { en: '{n} notes', pt: '{n} observações' },

  // ---- results: headline cards
  'kpi.reliabilityAt': { en: 'Reliability at {m}', pt: 'Confiabilidade em {m}' },
  'kpi.interval': { en: '90% interval {a} – {b}', pt: 'intervalo de 90% {a} – {b}' },
  'kpi.medianInterval': { en: 'median · 90% {a} – {b}', pt: 'mediana · 90% {a} – {b}' },
  'kpi.theMission': { en: 'the mission', pt: 'a missão' },
  'kpi.nDemands': { en: '{n} demands', pt: '{n} demandas' },
  'life.mttf': { en: 'Mean time to failure', pt: 'Tempo médio até a falha' },
  'life.mttfHint': { en: 'MTTF', pt: 'MTTF' },
  'life.b10': { en: 'B10 life', pt: 'Vida B10' },
  'life.b10Hint': { en: 'age by which 10% have failed', pt: 'idade em que 10% já falharam' },
  'life.mdtf': { en: 'Mean demands to failure', pt: 'Média de demandas até a falha' },
  'life.mdtfHint': { en: '1 / prob', pt: '1 / prob' },
  'read.wearout': {
    en: 'wear-out: failures become more frequent with age',
    pt: 'desgaste: as falhas ficam mais frequentes com a idade',
  },
  'read.early': {
    en: 'early failures: the hazard falls with age',
    pt: 'falhas prematuras: a taxa de falha cai com a idade',
  },
  'read.unclear': {
    en: 'the data cannot tell whether the hazard rises or falls',
    pt: 'os dados não dizem se a taxa de falha cresce ou cai',
  },
  'read.pShape': { en: 'P(shape > 1) {p}: {reading}', pt: 'P(shape > 1) {p}: {reading}' },
  'read.prob': { en: 'about 1 failure in {n} demands', pt: 'cerca de 1 falha a cada {n} demandas' },
  'read.rate': {
    en: 'one failure every {x} {unit}, typically',
    pt: 'uma falha a cada {x} {unit}, tipicamente',
  },

  // ---- results: prior versus posterior
  'pp.loading': { en: 'Loading the draws…', pt: 'Carregando as amostras…' },
  'pp.title': { en: 'What the data changed', pt: 'O que os dados mudaram' },
  'pp.intro': {
    en: "Each parameter's posterior (filled) over the prior that was in force (dashed), which is the prior after your bounds. A posterior much narrower than the prior means the data decided; one that copies the prior means they could not.",
    pt: 'A posteriori de cada parâmetro (preenchida) sobre a priori em vigor (tracejada), que é a priori depois dos seus limites. Uma posteriori muito mais estreita que a priori quer dizer que os dados decidiram; uma que repete a priori quer dizer que não conseguiram.',
  },
  'pp.posterior': { en: 'posterior', pt: 'posteriori' },
  'pp.prior': { en: 'prior ({label})', pt: 'priori ({label})' },
  'pp.priorScaled': { en: 'prior × {k}', pt: 'priori × {k}' },
  'pp.priorShort': { en: 'prior', pt: 'priori' },
  'pp.describe': { en: 'Prior and posterior of {name}', pt: 'Priori e posteriori de {name}' },
  'pp.priorSummary': {
    en: 'prior median {m} (90% {a} – {b}) →',
    pt: 'priori: mediana {m} (90% {a} – {b}) →',
  },
  'pp.postSummary': { en: 'posterior median {m} (90% {i})', pt: 'posteriori: mediana {m} (90% {i})' },
  'pp.narrower': { en: 'interval {k}× narrower', pt: 'intervalo {k}× mais estreito' },
  'pp.scaled': { en: 'prior drawn ×{k} to show its shape', pt: 'priori desenhada ×{k} para mostrar a forma' },

  // ---- results: table
  'tbl.quantity': { en: 'quantity', pt: 'grandeza' },
  'tbl.mean': { en: 'mean', pt: 'média' },
  'tbl.median': { en: 'median', pt: 'mediana' },
  'tbl.interval': { en: '90% interval', pt: 'intervalo de 90%' },
  'tbl.sd': { en: 'sd', pt: 'dp' },
  'tbl.rhatHint': {
    en: 'Potential scale reduction: should be below 1.01',
    pt: 'Redução potencial de escala: deve ficar abaixo de 1,01',
  },
  'tbl.essHint': {
    en: 'Effective sample size, bulk / tail: 400 or more is comfortable',
    pt: 'Tamanho efetivo de amostra, centro / cauda: 400 ou mais é confortável',
  },
  'tbl.reliability': { en: 'Reliability', pt: 'Confiabilidade' },
  'tbl.parameters': { en: 'Parameters', pt: 'Parâmetros' },
  'tbl.life': { en: 'Life', pt: 'Vida' },
  'tbl.lifeUnit': { en: 'Life ({unit})', pt: 'Vida ({unit})' },
  'tbl.other': { en: 'Other', pt: 'Outros' },
  'tbl.note': {
    en: 'Intervals are equal-tailed: 5% of the posterior lies below, 5% above. For reliability the mean is the probability of surviving given everything observed; for skewed quantities such as life, read the median.',
    pt: 'Os intervalos deixam 5% da posteriori abaixo e 5% acima. Para a confiabilidade, a média é a probabilidade de sobreviver dado tudo o que foi observado; para grandezas assimétricas, como a vida, leia a mediana.',
  },
  'tbl.dropped': {
    en: 'Not shown because every draw was infinite: {list}.',
    pt: 'Fora da tabela porque todas as amostras foram infinitas: {list}.',
  },

  // ---- results: diagnostics
  'diag.how': { en: 'How the sampler explored the posterior.', pt: 'Como o amostrador percorreu a posteriori.' },
  'diag.aria': { en: 'Diagnostic view', pt: 'Visão de diagnóstico' },
  'diag.Traces': { en: 'Traces', pt: 'Traços' },
  'diag.Marginals': { en: 'Marginals', pt: 'Marginais' },
  'diag.Joint': { en: 'Joint', pt: 'Conjunta' },
  'marg.logScale': { en: '{label} (log scale)', pt: '{label} (escala log)' },
  'marg.density': { en: 'density', pt: 'densidade' },
  'marg.densityPerDecade': { en: 'density per decade', pt: 'densidade por década' },
  'marg.median': { en: 'median', pt: 'mediana' },
  'marg.describe': { en: 'Posterior distribution of {label}', pt: 'Distribuição a posteriori de {label}' },
  'marg.caption': {
    en: 'median {m} · 90% interval {a} to {b}',
    pt: 'mediana {m} · intervalo de 90% de {a} a {b}',
  },
  'marg.clipped': { en: 'axis clipped: the tail runs out to {x}', pt: 'eixo cortado: a cauda vai até {x}' },
  'trace.draw': { en: 'draw', pt: 'amostra' },
  'trace.all': { en: 'all draws', pt: 'todas as amostras' },
  'trace.chain': { en: 'chain {n}', pt: 'cadeia {n}' },
  'trace.describe': {
    en: 'Sampler trace for {name}, one line per chain',
    pt: 'Traço do amostrador para {name}, uma linha por cadeia',
  },
  'trace.caption': {
    en: 'Chains should overlap and look like noise around a stable level. A chain drifting apart from the others means they have not agreed.',
    pt: 'As cadeias devem se sobrepor e parecer ruído em torno de um nível estável. Uma cadeia que se afasta das outras indica que elas não concordaram.',
  },
  'joint.single': {
    en: 'This model has a single parameter, so there is no joint distribution to show. See Marginals instead.',
    pt: 'Este modelo tem um único parâmetro, então não há distribuição conjunta. Veja as marginais.',
  },
  'joint.intro': {
    en: 'Posterior density over each pair of parameters.',
    pt: 'Densidade a posteriori de cada par de parâmetros.',
  },
  'joint.aria': { en: 'How to draw the joint density', pt: 'Como desenhar a densidade conjunta' },
  'joint.3d': { en: '3D surface', pt: 'Superfície 3D' },
  'joint.2d': { en: '2D contour', pt: 'Contorno 2D' },
  'joint.vs': { en: '{x} vs {y}', pt: '{x} × {y}' },
  'joint.notEnough': {
    en: 'Not enough draws to estimate a density.',
    pt: 'Amostras insuficientes para estimar uma densidade.',
  },
  'joint.caption': {
    en: 'A ridge running diagonally means the data cannot tell these two apart — only their combination is pinned down.',
    pt: 'Uma crista na diagonal indica que os dados não separam os dois: só a combinação deles fica determinada.',
  },
  'joint.rotate': {
    en: 'Drag to rotate; the shadow underneath is the same density seen from above.',
    pt: 'Arraste para girar; a sombra embaixo é a mesma densidade vista de cima.',
  },
  'joint.describe': {
    en: 'Joint posterior density of {x} and {y}',
    pt: 'Densidade conjunta a posteriori de {x} e {y}',
  },
  'joint.draws': { en: 'draws', pt: 'amostras' },

  // ---- results: reliability curve
  'rel.title': { en: 'Reliability', pt: 'Confiabilidade' },
  'rel.titleTime': { en: 'Reliability over time', pt: 'Confiabilidade ao longo do tempo' },
  'rel.titleDemands': { en: 'Reliability over demands', pt: 'Confiabilidade ao longo das demandas' },
  'rel.none': {
    en: 'There is no reliability curve for this run: it had neither a mission nor any data to set a time scale by. Set a mission time on the Model tab and run again.',
    pt: 'Esta rodada não tem curva de confiabilidade: não havia missão nem dados para definir a escala de tempo. Defina um tempo de missão na aba Modelo e rode de novo.',
  },
  'rel.time': { en: 'time', pt: 'tempo' },
  'rel.demands': { en: 'demands survived', pt: 'demandas sobrevividas' },
  'rel.axis': { en: 'reliability R', pt: 'confiabilidade R' },
  'rel.axisFrom': { en: 'reliability R (axis from {f})', pt: 'confiabilidade R (eixo a partir de {f})' },
  'rel.band': { en: '90% interval', pt: 'intervalo de 90%' },
  'rel.median': { en: 'median', pt: 'mediana' },
  'rel.mean': { en: 'mean', pt: 'média' },
  'rel.mission': { en: 'mission', pt: 'missão' },
  'rel.missions': { en: 'mission times', pt: 'tempos de missão' },
  'rel.atMission': { en: 'at the mission', pt: 'na missão' },
  'rel.km': { en: 'data (Kaplan–Meier)', pt: 'dados (Kaplan–Meier)' },
  'rel.censored': { en: 'censored', pt: 'censurado' },
  'rel.describe': {
    en: 'Posterior reliability against time, with a 90% credible band',
    pt: 'Confiabilidade a posteriori ao longo do tempo, com faixa de 90%',
  },
  'rel.caption': {
    en: 'The line is the posterior mean, the probability of surviving that long given everything observed; the band holds 90% of the posterior at each point, so it widens where the data stop saying much. Red marks the mission.',
    pt: 'A linha é a média a posteriori, a probabilidade de sobreviver até ali dado tudo o que foi observado; a faixa contém 90% da posteriori em cada ponto, então ela abre onde os dados deixam de informar. O vermelho marca a missão.',
  },
  'rel.captionKm': {
    en: 'The black steps are the data alone (Kaplan–Meier, ticks for censored units): the fitted curve should run through them.',
    pt: 'Os degraus pretos são só os dados (Kaplan–Meier, com traços nas unidades censuradas): a curva ajustada deve passar por eles.',
  },
} satisfies Record<string, Record<Lang, string>>

export type Key = keyof typeof STRINGS

let numberLang: Lang = 'en'

/** Write a formatted number the way the current language does. */
export function decimal(text: string): string {
  return numberLang === 'pt' ? text.replace(/\./g, ',') : text
}

/** The locale tag for `toLocaleString`, so 10000 reads 10,000 or 10.000. */
export function numberLocale(): string {
  return numberLang === 'pt' ? 'pt-BR' : 'en'
}

export function plotSeparators(): string {
  return numberLang === 'pt' ? ',.' : '.,'
}

function fill(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template
  return template.replace(/\{(\w+)\}/g, (match, name: string) =>
    name in vars ? String(vars[name]) : match,
  )
}

/**
 * A translation with React nodes in its placeholders, for sentences that
 * carry markup (`<code>failure = 1</code>` in the middle of a phrase).
 */
export function rich(template: string, nodes: Record<string, ReactNode>): ReactNode[] {
  return template.split(/(\{\w+\})/).map((part, index) => {
    const match = /^\{(\w+)\}$/.exec(part)
    return match && match[1] in nodes ? <span key={index}>{nodes[match[1]]}</span> : part
  })
}

interface I18n {
  lang: Lang
  setLang: (lang: Lang) => void
  t: (key: Key, vars?: Record<string, string | number>) => string
}

const I18nContext = createContext<I18n | null>(null)

const STORAGE_KEY = 'pirao.lang'

function initialLang(): Lang {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    if (stored === 'pt' || stored === 'en') return stored
  } catch {
    /* storage can be unavailable; fall through to the browser */
  }
  // As in farofa: follow the browser, and default to Portuguese.
  return navigator.language?.toLowerCase().startsWith('en') ? 'en' : 'pt'
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang)

  // Set before the children render, so every formatter they call agrees.
  numberLang = lang
  document.documentElement.lang = lang === 'pt' ? 'pt-BR' : 'en'

  const setLang = useCallback((next: Lang) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, next)
    } catch {
      /* not worth failing over */
    }
    setLangState(next)
  }, [])

  const value = useMemo<I18n>(
    () => ({
      lang,
      setLang,
      // An unknown key (a stage name the server added later) shows as itself.
      t: (key, vars) => fill(STRINGS[key]?.[lang] ?? key, vars),
    }),
    [lang, setLang],
  )
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18n {
  const context = useContext(I18nContext)
  if (!context) throw new Error('useI18n outside I18nProvider')
  return context
}
