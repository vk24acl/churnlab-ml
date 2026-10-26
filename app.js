/* Pure inference exported for both browser use and Node parity checks. */
function predict(x,m){
  if(x.length!==m.features.length||x.some(v=>!Number.isFinite(v)))throw Error('Six finite features required');
  const contributions=x.map((v,i)=>(v-m.mean[i])/m.scale[i]*m.coefficients[i]);
  const raw=m.intercept+contributions.reduce((a,b)=>a+b,0);
  const clipped=Math.max(-13.815509557963773,Math.min(13.815509557935018,raw));
  return {probability:1/(1+Math.exp(-(clipped*m.calibration_coefficient+m.calibration_intercept))),contributions};
}
if(typeof module!=='undefined')module.exports={predict};
if(typeof document!=='undefined'){
  const pct=x=>(100*x).toFixed(1)+'%';
  const r=window.REPORT,m=window.MODEL;
  document.getElementById('metrics').innerHTML=[['Test average precision',r.test.average_precision.toFixed(3)],['ROC AUC',r.test.roc_auc.toFixed(3)],['Top-decile lift',r.test.targeting_10pct.lift.toFixed(2)+'×'],['Brier score',r.test.brier.toFixed(3)]].map(([a,b])=>`<div class="metric"><small>${a}</small><strong>${b}</strong></div>`).join('');
  const table=(headers,rows)=>'<table><thead><tr>'+headers.map(h=>`<th>${h}</th>`).join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(c=>`<td>${c}</td>`).join('')+'</tr>').join('')+'</tbody></table>';
  document.getElementById('comparison').innerHTML=table(['Candidate','Validation AP'],Object.entries(r.candidates).map(([n,v])=>[n.replaceAll('_',' '),v.validation_ap.toFixed(3)]));
  document.getElementById('importance').innerHTML=r.permutation_importance.sort((a,b)=>b.ap_drop_mean-a.ap_drop_mean).map(a=>`<div class="bar"><span>${a.feature.replaceAll('_',' ')}</span><i style="width:${Math.max(0,a.ap_drop_mean)*500}px"></i><b>${a.ap_drop_mean.toFixed(3)}</b></div>`).join('');
  const capacity=document.getElementById('capacity');
  function capacityUpdate(){const c=r.capacity_curve[+capacity.value];document.getElementById('capacity-result').textContent=`Contact ${c.capacity} of 900 test customers (${pct(c.fraction)}): precision ${pct(c.precision)}, recall ${pct(c.recall)}, lift ${c.lift.toFixed(2)}×.`;}
  capacity.oninput=capacityUpdate;capacityUpdate();
  document.getElementById('calibration').innerHTML=table(['Customers','Mean forecast','Observed churn'],r.calibration.map(a=>[a.count,pct(a.predicted),pct(a.observed)]));
  if(m){
    const form=document.getElementById('scenario');
    const defaults=[12,4,2,60,1,-.2],ranges=[[1,60,1],[0,25,.5],[0,8,1],[15,120,1],[0,1,1],[-1,1,.05]];
    form.innerHTML=m.features.map((f,i)=>`<label>${f.replaceAll('_',' ')}<input aria-label="${f}" type="number" min="${ranges[i][0]}" max="${ranges[i][1]}" step="${ranges[i][2]}" value="${defaults[i]}"></label>`).join('');
    function update(){const inputs=[...form.querySelectorAll('input')];if(inputs.some(e=>!e.checkValidity())){document.getElementById('prediction').textContent='Use values within the displayed ranges.';return;} const out=predict(inputs.map(e=>+e.value),m);document.getElementById('prediction').textContent=`Estimated simulated churn probability: ${pct(out.probability)}. Validation decision threshold: ${pct(m.threshold)}.`;}
    form.oninput=update;update();
  } else document.getElementById('scenario').textContent='Interactive inference export is available only when logistic regression wins validation.';
}
