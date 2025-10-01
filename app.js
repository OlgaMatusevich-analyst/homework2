/* =====================================================================
   Titanic Binary Classifier — TF.js, no server
   - Robust CSV parsing with Papa
   - Preprocessing: impute, standardize, one-hot, engineered features
   - ROC/AUC on validation set (custom canvas renderer)
   - Confusion matrix + Precision/Recall/F1 with threshold slider
   - Predict/Export for Kaggle (PassengerId, Survived)
===================================================================== */

// --------------------------- Globals & schema ------------------------
const SCHEMA = {
  target: 'Survived',
  id: 'PassengerId',
  featuresCore: ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'],
  numeric: ['Age','Fare'],
  categorical: {
    Sex: ['male','female'],
    Pclass: ['1','2','3'],
    Embarked: ['C','Q','S']
  }
};

// raw rows from CSV
let TRAIN_ROWS = [];
let TEST_ROWS  = [];

// After preprocessing
let FEATURE_NAMES = [];     // final feature names (train order = test order)
let TRAIN_X, TRAIN_y;       // tensors
let VAL_X, VAL_y;           // tensors
let VAL_TRUE = [], VAL_PROB = []; // arrays for ROC/metrics

// Stats to reuse on test preprocessing
let STATS = {
  mean: { Age: null, Fare: null },
  std:  { Age: null, Fare: null }
};
let ONEHOT_MAP = {};        // { catName -> [values...] }
let ADD_FAMILY = true;
let MODEL;

// ------------------------------ Utils --------------------------------
const $ = (id) => document.getElementById(id);
const setHTML = (id, html) => { $(id).innerHTML = html; };

function alertError(msg, err) {
  console.error(msg, err||'');
  alert(msg);
}

function toNumber(v) {
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  if (s === '' || s.toLowerCase() === 'nan') return null;
  const x = Number(s);
  return Number.isFinite(x) ? x : null;
}

function computeMedian(arr) {
  const xs = arr.filter(x => x !== null && x !== undefined).sort((a,b)=>a-b);
  if (!xs.length) return 0;
  const m = Math.floor(xs.length/2);
  return xs.length%2 ? xs[m] : (xs[m-1]+xs[m])/2;
}
function computeMode(arr) {
  const cnt = new Map();
  for (const x of arr) {
    const k = (x===null||x===undefined||x==='')?'NA':String(x);
    cnt.set(k,(cnt.get(k)||0)+1);
  }
  let best='NA',mx=-1;
  cnt.forEach((v,k)=>{ if(v>mx){mx=v;best=k;} });
  return best==='NA'? null : best;
}

function renderTablePreview(rows, limit=10) {
  if (!rows.length) return '';
  const cols = Object.keys(rows[0]);
  const head = `<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`;
  const body = rows.slice(0,limit)
    .map(r=>`<tr>${cols.map(c=>`<td>${String(r[c]??'')}</td>`).join('')}</tr>`).join('');
  return `<table style="border-collapse:collapse;">
    <thead style="background:#f3f4f6;">${head}</thead>
    <tbody>${body}</tbody>
  </table>
  <style>
    #tableBox table { font-size:13px; }
    #tableBox th,#tableBox td { border:1px solid #e5e7eb; padding:4px 8px; }
    #tableBox th { position:sticky; top:0; }
  </style>`;
}

// Simple bar renderer (value array 0..1)
function renderBars(canvasId, labels, values) {
  const cv = $(canvasId); if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0,0,W,H);
  const pad=18, unit = (H - pad*2) / labels.length;
  for (let i=0;i<labels.length;i++) {
    const y = pad + i*unit + unit/4;
    const barW = (W - 120) * (values[i]??0);
    // background
    ctx.fillStyle = '#e5e7eb';
    ctx.fillRect(100, y, W-120, unit/2);
    // value bar
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(100, y, barW, unit/2);
    // labels
    ctx.fillStyle = '#111827';
    ctx.font = '12px system-ui';
    ctx.fillText(labels[i], 10, y+unit/2-2);
    ctx.fillText((values[i]??0).toFixed(3), W-50, y+unit/2-2);
  }
}

// --------------------------- CSV loading ------------------------------
function parseCsvFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      transform: (v) => v, // не трогаем, чистим позже
      complete: (res) => resolve(res.data),
      error: (err) => reject(err)
    });
  });
}

$('trainFile').addEventListener('change', async (e)=>{
  try {
    TRAIN_ROWS = await parseCsvFile(e.target.files[0]);
    setHTML('tableBox', renderTablePreview(TRAIN_ROWS, 10));
  } catch(err){ alertError('Failed to parse train.csv', err); }
});
$('testFile').addEventListener('change', async (e)=>{
  try {
    TEST_ROWS = await parseCsvFile(e.target.files[0]);
  } catch(err){ alertError('Failed to parse test.csv', err); }
});

// ------------------------------ EDA -----------------------------------
function survivalRateBy(rows, key) {
  const counts = {}; // key -> {n: total, s: survived}
  for (const r of rows) {
    const y = Number(r[SCHEMA.target]);
    if (!Number.isFinite(y)) continue; // на train
    const k = r[key] ?? 'NA';
    if (!counts[k]) counts[k] = {n:0, s:0};
    counts[k].n++;
    counts[k].s += y ? 1 : 0;
  }
  const labels = Object.keys(counts);
  const values = labels.map(k => counts[k].n ? counts[k].s / counts[k].n : 0);
  return {labels, values};
}

$('btnInspect').addEventListener('click', ()=>{
  if (!TRAIN_ROWS.length) return alert('Load train.csv first');
  // shape + missing %
  const cols = Object.keys(TRAIN_ROWS[0]);
  const miss = {};
  for (const c of cols) {
    let m=0;
    for (const r of TRAIN_ROWS) {
      const v = r[c];
      if (v===undefined || v===null || String(v).trim()==='' ) m++;
    }
    miss[c] = m/TRAIN_ROWS.length;
  }
  const shape = `${TRAIN_ROWS.length} rows × ${cols.length} cols`;
  setHTML('previewInfo', `Shape: ${shape}\nMissing %: ${JSON.stringify(miss, null, 2)}`);

  // bars
  const bySex   = survivalRateBy(TRAIN_ROWS,'Sex');
  const byClass = survivalRateBy(TRAIN_ROWS,'Pclass');
  renderBars('barSex', bySex.labels, bySex.values);
  renderBars('barClass', byClass.labels, byClass.values);
});

// --------------------------- Preprocessing ----------------------------

// Build one-hot domain from train
function buildFeatureSpace(trainRows, addFamily) {
  const cats = JSON.parse(JSON.stringify(SCHEMA.categorical)); // clone defaults
  // if unseen categories exist in train — add them
  for (const r of trainRows) {
    ['Sex','Pclass','Embarked'].forEach(k=>{
      const v = String(r[k]??'').trim();
      if (!v) return;
      if (!cats[k].includes(v)) cats[k].push(v);
    });
  }
  ONEHOT_MAP = cats;

  // stats for numeric z-score
  const ages = trainRows.map(r => toNumber(r.Age)).filter(x=>x!==null);
  const fares= trainRows.map(r => toNumber(r.Fare)).filter(x=>x!==null);
  const ageMed  = computeMedian(ages);
  const fareMed = computeMedian(fares);
  const ageArr  = trainRows.map(r => toNumber(r.Age)  ?? ageMed);
  const fareArr = trainRows.map(r => toNumber(r.Fare) ?? fareMed);

  const ageMean = ageArr.reduce((a,b)=>a+b,0)/ageArr.length;
  const fareMean= fareArr.reduce((a,b)=>a+b,0)/fareArr.length;
  const ageStd  = Math.sqrt(ageArr.reduce((s,x)=>s+(x-ageMean)**2,0)/ageArr.length) || 1;
  const fareStd = Math.sqrt(fareArr.reduce((s,x)=>s+(x-fareMean)**2,0)/fareArr.length) || 1;

  STATS.mean.Age  = ageMean; STATS.std.Age  = ageStd;
  STATS.mean.Fare = fareMean; STATS.std.Fare = fareStd;

  // final feature names (order!)
  const names = [
    'Age_z','Fare_z',
    ...cats.Sex.map(v=>`Sex=${v}`),
    ...cats.Pclass.map(v=>`Pclass=${v}`),
    ...cats.Embarked.map(v=>`Embarked=${v}`)
  ];
  if (addFamily) names.push('FamilySize','IsAlone');
  else names.push('SibSp','Parch');

  FEATURE_NAMES = names;
  return cats;
}

function preprocessRows(rows, cats, addFamily, isTrain) {
  // imputations
  const ageMed   = computeMedian(rows.map(r=>toNumber(r.Age)));
  const embMode  = computeMode(rows.map(r=>r.Embarked));
  const fareMed  = computeMedian(rows.map(r=>toNumber(r.Fare)));

  const X = [];
  const Y = [];

  for (const r of rows) {
    // base values
    const Age  = toNumber(r.Age)  ?? ageMed;
    const Fare = toNumber(r.Fare) ?? fareMed;
    const Sex  = String(r.Sex??'').trim();
    const Pclass = String(r.Pclass??'').trim();
    const Embarked = String(r.Embarked??embMode??'').trim();

    // z-scores by train stats
    const age_z  = (Age  - STATS.mean.Age )/ STATS.std.Age;
    const fare_z = (Fare - STATS.mean.Fare)/ STATS.std.Fare;

    const feats = [];
    feats.push(age_z, fare_z);

    // one-hot
    for (const v of cats.Sex)     feats.push(Sex===v ? 1 : 0);
    for (const v of cats.Pclass)  feats.push(Pclass===v ? 1 : 0);
    for (const v of cats.Embarked)feats.push(Embarked===v ? 1 : 0);

    // engineered OR raw
    const sib = toNumber(r.SibSp) || 0;
    const par = toNumber(r.Parch) || 0;
    if (addFamily) {
      const fam = sib + par + 1;
      feats.push(fam, fam===1 ? 1 : 0);
    } else {
      feats.push(sib, par);
    }

    X.push(feats);

    if (isTrain) {
      const y = Number(r[SCHEMA.target]);
      if (Number.isFinite(y)) Y.push(y);
      else Y.push(0);
    }
  }

  return { X, Y };
}

$('btnPreprocess').addEventListener('click', ()=>{
  if (!TRAIN_ROWS.length) return alert('Load train.csv first');
  ADD_FAMILY = $('toggleFamily').checked;

  const cats = buildFeatureSpace(TRAIN_ROWS, ADD_FAMILY);
  const pre  = preprocessRows(TRAIN_ROWS, cats, ADD_FAMILY, true);

  // tensors + stratified split 80/20
  const {trainIdx, valIdx} = stratifiedSplit(pre.Y, 0.2, 42);
  const Xtr = trainIdx.map(i => pre.X[i]);
  const ytr = trainIdx.map(i => pre.Y[i]);
  const Xva = valIdx.map(i => pre.X[i]);
  const yva = valIdx.map(i => pre.Y[i]);

  TRAIN_X = tf.tensor2d(Xtr); TRAIN_y = tf.tensor2d(ytr, [ytr.length,1]);
  VAL_X   = tf.tensor2d(Xva); VAL_y   = tf.tensor2d(yva, [yva.length,1]);

  // info boxes
  setHTML('featInfo', `Features (${FEATURE_NAMES.length}):\n${FEATURE_NAMES.join(', ')}`);
  setHTML('shapeInfo',
`X shape: ${pre.X.length},${FEATURE_NAMES.length}
Y shape: ${pre.Y.length},1
Standardization: {
  "mean": {
    "Age": ${STATS.mean.Age},
    "Fare": ${STATS.mean.Fare}
  },
  "std": {
    "Age": ${STATS.std.Age},
    "Fare": ${STATS.std.Fare}
  }
}`);

  setHTML('predInfo','');
});

// --------------------------- Stratified split -------------------------
function stratifiedSplit(y, valRatio=0.2, seed=123) {
  const pos=[], neg=[];
  y.forEach((v,i)=> (v?pos:neg).push(i));
  function shuffle(a){
    let r = seed;
    for (let i=a.length-1;i>0;i--){
      r = (1664525*r + 1013904223) % 4294967296;
      const j = Math.floor((r/4294967296)*(i+1));
      [a[i],a[j]] = [a[j],a[i]];
    }
    return a;
  }
  shuffle(pos); shuffle(neg);
  const vp = Math.floor(pos.length*valRatio);
  const vn = Math.floor(neg.length*valRatio);
  const val = pos.slice(0,vp).concat(neg.slice(0,vn));
  const train = pos.slice(vp).concat(neg.slice(vn));
  return {trainIdx: shuffle(train), valIdx: shuffle(val)};
}

// ------------------------------ Model --------------------------------
$('btnBuild').addEventListener('click', ()=>{
  if (!TRAIN_X) return alert('Run preprocessing first');

  MODEL = tf.sequential();
  MODEL.add(tf.layers.dense({units:16, activation:'relu', inputShape:[FEATURE_NAMES.length]}));
  MODEL.add(tf.layers.dense({units:1, activation:'sigmoid'}));
  MODEL.compile({optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy']});
  alert('Model built');
});

$('btnSummary').addEventListener('click', ()=>{
  if (!MODEL) return alert('Build model first');
  const lines = [];
  MODEL.summary(80, undefined, (s)=>lines.push(s));
  setHTML('modelSummary', lines.join('\n'));
});

// ------------------------------ Training ------------------------------
$('btnTrain').addEventListener('click', async ()=>{
  try{
    if (!MODEL) return alert('Build model first');
    if (!TRAIN_X || !VAL_X) return alert('Run preprocessing first');

    // train
    await MODEL.fit(TRAIN_X, TRAIN_y, {
      epochs:50,
      batchSize:32,
      validationData:[VAL_X, VAL_y],
      verbose:0,
      callbacks: tf.callbacks.earlyStopping({ monitor:'val_loss', patience:5, restoreBestWeight:true })
    });

    // evaluate on validation
    const probsT = await MODEL.predict(VAL_X);
    VAL_PROB = Array.from(await probsT.data());
    VAL_TRUE = Array.from(await VAL_y.data());
    probsT.dispose();

    // ROC/AUC
    const roc = computeROC(VAL_TRUE, VAL_PROB);
    const auc = aucFromROC(roc);
    renderROC('rocCanvas', roc, auc);

    // metrics @ current threshold
    const thr = parseFloat($('threshold').value);
    updateMetrics(thr);

  }catch(err){ alertError('Training failed', err); }
});

// when slider moves
$('threshold').addEventListener('input', ()=>{
  if (!VAL_PROB.length) return;
  updateMetrics(parseFloat($('threshold').value));
});

// ------------------------ Metrics/Confusion ---------------------------
function confusionAt(thr, yTrue, yProb) {
  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0;i<yTrue.length;i++){
    const y = yTrue[i] ? 1:0;
    const p = (yProb[i] >= thr) ? 1:0;
    if (y===1 && p===1) tp++;
    else if (y===0 && p===0) tn++;
    else if (y===0 && p===1) fp++;
    else fn++;
  }
  const precision = tp+fp ? tp/(tp+fp) : 0;
  const recall    = tp+fn ? tp/(tp+fn) : 0;
  const acc       = (tp+tn)/(tp+tn+fp+fn);
  const f1        = (precision+recall)? 2*precision*recall/(precision+recall) : 0;
  return {tp,tn,fp,fn, precision, recall, acc, f1};
}

function updateMetrics(thr) {
  const m = confusionAt(thr, VAL_TRUE, VAL_PROB);
  setHTML('metricsBox',
` ${thr.toFixed(2)}

 Confusion Matrix
 Actual+  Pred+ = ${m.tp}  |  Pred- = ${m.fn}
 Actual-  Pred+ = ${m.fp}  |  Pred- = ${m.tn}

 Accuracy=${m.acc.toFixed(3)}  Precision=${m.precision.toFixed(3)}  Recall=${m.recall.toFixed(3)}
 F1=${m.f1.toFixed(3)}
`);
}

// --------------------------- ROC / AUC --------------------------------
function computeROC(yTrue, yProb) {
  const pairs = yTrue.map((y,i)=>({y:y?1:0, p:Number(yProb[i])||0}));
  pairs.sort((a,b)=> b.p - a.p);
  const P = pairs.reduce((s,r)=> s + r.y, 0);
  const N = pairs.length - P;
  if (P===0 || N===0) return [{fpr:0,tpr:0},{fpr:1,tpr:1}];

  let tp=0, fp=0, i=0; const roc=[{fpr:0,tpr:0}];
  while (i<pairs.length){
    const thr = pairs[i].p;
    while (i<pairs.length && pairs[i].p===thr){
      if (pairs[i].y) tp++; else fp++;
      i++;
    }
    roc.push({ fpr: fp/N, tpr: tp/P });
  }
  roc.push({fpr:1,tpr:1});
  return roc;
}
function aucFromROC(roc) {
  let area=0;
  for (let i=1;i<roc.length;i++){
    const x1=roc[i-1].fpr, y1=roc[i-1].tpr;
    const x2=roc[i].fpr,   y2=roc[i].tpr;
    area += (x2-x1)*(y1+y2)/2;
  }
  return area;
}
function renderROC(canvasId, roc, auc) {
  const cvs=$(canvasId); if(!cvs) return; const ctx=cvs.getContext('2d');
  const W=cvs.width, H=cvs.height, pad=34;
  ctx.clearRect(0,0,W,H);
  // frame
  ctx.strokeStyle='#e5e7eb'; ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
  // diagonal
  ctx.beginPath(); ctx.strokeStyle='#cbd5e1';
  ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,pad); ctx.stroke();
  // curve
  ctx.beginPath(); ctx.strokeStyle='#3b82f6'; ctx.lineWidth=2;
  for(let i=0;i<roc.length;i++){
    const x = pad + (W-2*pad)*roc[i].fpr;
    const y = H - pad - (H-2*pad)*roc[i].tpr;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
  // labels
  ctx.fillStyle='#111827'; ctx.font='12px system-ui';
  ctx.fillText(`AUC = ${auc.toFixed(3)}`, pad+6, pad+14);
  ctx.fillText('FPR', W/2 - 10, H-8);
  ctx.save(); ctx.translate(12, H/2 + 10); ctx.rotate(-Math.PI/2); ctx.fillText('TPR', 0, 0); ctx.restore();
}

// ----------------------- Prediction / Export --------------------------
$('btnPredict').addEventListener('click', async ()=>{
  try{
    if (!MODEL || !TEST_ROWS.length) return alert('Need model and test.csv');
    const cats = ONEHOT_MAP;
    const pre  = preprocessRows(TEST_ROWS, cats, ADD_FAMILY, false);

    const Xtest = tf.tensor2d(pre.X);
    const probT = await MODEL.predict(Xtest);
    const probs = Array.from(await probT.data());
    probT.dispose(); Xtest.dispose();

    const thr = parseFloat($('threshold').value);
    const preds = probs.map(p => p >= thr ? 1 : 0);

    const rows = TEST_ROWS.map((r,i)=>({ PassengerId:r[SCHEMA.id], Survived: preds[i] }));
    const rowsProb = TEST_ROWS.map((r,i)=>({ PassengerId:r[SCHEMA.id], Prob: probs[i] }));

    window.__submission = rows;
    window.__probabilities = rowsProb;
    setHTML('predInfo', `Predicted ${rows.length} rows. Threshold=${thr.toFixed(2)}. Ready to export.`);
  }catch(err){ alertError('Prediction failed', err); }
});

$('btnExport').addEventListener('click', ()=>{
  if (!window.__submission) return alert('Run Predict first');
  downloadCsv('submission.csv', window.__submission);
  downloadCsv('probabilities.csv', window.__probabilities);
});

$('btnSaveModel').addEventListener('click', async ()=>{
  if (!MODEL) return alert('No model to save');
  await MODEL.save('downloads://titanic-tfjs');
});

function downloadCsv(name, rows) {
  const cols = Object.keys(rows[0]);
  const lines = [cols.join(',')].concat(rows.map(r => cols.map(c => r[c]).join(',')));
  const blob = new Blob(['\uFEFF' + lines.join('\n')], {type:'text/csv;charset=utf-8;'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  URL.revokeObjectURL(a.href);
}
