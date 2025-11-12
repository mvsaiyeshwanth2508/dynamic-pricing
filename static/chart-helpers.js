// chart-helpers.js
function plotTimeSeries(ctxId, data) {
  const labels = data.map(d => d.Date);
  const vals = data.map(d => d.Revenue);
  const ctx = document.getElementById(ctxId).getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ label: 'Revenue', data: vals, fill:false }] },
    options: {}
  });
}

function plotCategory(ctxId, data) {
  const labels = data.map(d => d.Product_Category);
  const vals = data.map(d => d.Revenue);
  const ctx = document.getElementById(ctxId).getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets:[{ label:'Revenue', data: vals}] },
    options: {}
  });
}

function plotScatter(ctxId, data) {
  const points = data.map(d => ({x: d.Price_Set, y: d.Sales_Volume, r: Math.min(10, Math.max(3, d.Revenue/100))}));
  const ctx = document.getElementById(ctxId).getContext('2d');
  new Chart(ctx, {
    type: 'bubble',
    data: { datasets: [{ label:'Price vs Volume', data: points }]},
    options: {}
  });
}

function plotDiscount(ctxId, data) {
  const labels = data.map(d=>d.Discount_Bucket);
  const vals = data.map(d=>d.Sales_Volume);
  const ctx = document.getElementById(ctxId).getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets:[{ label:'Avg Sales Volume', data: vals }]},
    options: {}
  });
}

// on page load
document.addEventListener('DOMContentLoaded', function(){
  if (typeof report !== 'undefined') {
    if (report.time_series) plotTimeSeries('tsChart', report.time_series);
    if (report.by_category) plotCategory('catChart', report.by_category);
    if (report.scatter) plotScatter('scatterChart', report.scatter);
    if (report.discount_effect) plotDiscount('discChart', report.discount_effect);
  }
});
