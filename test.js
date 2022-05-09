function sortArr(a, b) {
  let out = [];
  let i = 0;
  let j = 0;
  console.log(b + 'Starting' + a);
  for (let k = 0; k < n.length; k++) {
    if (a[i] < b[j]) {
      out[k] = a[i];
      console.log(`putting out[${k}] = a[${i}] or ${a[i]}`);
      i++;
    } else {
      out[k] = b[j];
      console.log(`putting out[${k}] = b[${j}] or ${b[j]}`);
      j++;
    }
  }
}
