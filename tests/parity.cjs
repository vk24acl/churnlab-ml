const assert=require('node:assert/strict'),fs=require('node:fs');
const {predict}=require('../app.js');
const m=JSON.parse(fs.readFileSync('reports/model.json')),f=JSON.parse(fs.readFileSync('reports/parity.json'));
f.x.forEach((x,i)=>assert.ok(Math.abs(predict(x,m).probability-f.probabilities[i])<1e-10));
assert.throws(()=>predict([NaN],m));
console.log('Browser/Python prediction parity passed for 10 held-out customer snapshots.');
