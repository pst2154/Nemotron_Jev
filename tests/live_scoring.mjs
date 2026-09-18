import assert from 'node:assert/strict';
const base = process.env.TEST_URL;
if (!base) throw new Error('Set TEST_URL');
const questions = {
  route: {type:'choice', instructions:'Which team should handle this? Payment and payout issues belong to billing, including failed payments.', criteria:{billing:'Payments, payouts, invoices or refunds',technical:'Application bugs or infrastructure outages unrelated to payments'}},
  urgent: {type:'noul', instructions:'Is the issue urgent and business blocking?'},
  severity: {type:'score', instructions:'How severe is the issue?', criteria:['Minor cosmetic inconvenience','Some functionality degraded','Critical business blocking failure']}
};
async function call(state, qs=questions) {
  const response=await fetch(base+'/v1/systemone',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'jev-latest',state,questions:qs})});
  const body=await response.json();
  assert.equal(response.status,200,JSON.stringify(body));
  assert.equal(body.scoring.method,'diffusion_masked_token_candidate_softmax');
  assert.deepEqual(Object.keys(body.answers),Object.keys(qs));
  for (const [id,a] of Object.entries(body.answers)) {
    if(a.type==='noul') { assert(a.noul>=0 && a.noul<=1); continue; }
    const ps=Object.values(a.probabilities);
    assert(ps.every(p=>Number.isFinite(p)&&p>=0&&p<=1));
    assert(Math.abs(ps.reduce((x,y)=>x+y,0)-1)<1e-5);
    assert.equal(a.confidence,Math.max(...ps));
    if(a.type==='score') assert(Math.abs(a.score-ps.reduce((s,p,i)=>s+i*p,0))<1e-5);
    else assert(Object.hasOwn(qs[id].criteria,a.choice));
  }
  return body;
}
const severe=await call('Customer payouts have failed for three days. Merchants cannot access their money. This is an urgent business-blocking failure.');
console.log('severe',JSON.stringify(severe));
assert.equal(severe.answers.route.choice,'billing');
assert(severe.answers.urgent.noul>.5);
const mild=await call('A cosmetic icon is one pixel off center in the app. Everything works normally. Not urgent.');
console.log('mild',JSON.stringify(mild));
assert.equal(mild.answers.route.choice,'technical');
assert(mild.answers.urgent.noul<.5);
assert(mild.answers.severity.score<severe.answers.severity.score);
const many=Object.fromEntries(Array.from({length:16},(_,i)=>['question '+i,{type:'noul',instructions:i%2?'Is 2 greater than 9?':'Is 9 greater than 2?'}]));
const batch=await call('Use the stated numbers.',many);
for(const [id,a] of Object.entries(batch.answers))assert(Number(id.split(' ')[1])%2?a.noul<.5:a.noul>.5);
console.log('16-question batch passed',batch.metrics);
console.log('ALL SCORING TESTS PASSED');
