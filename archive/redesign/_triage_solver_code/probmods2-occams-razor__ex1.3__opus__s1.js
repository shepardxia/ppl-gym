var maxNumber = 20;
var nums = _.range(1, maxNumber + 1);

// Build rule-based extensions
var multiples = map(function(n) {
  return {ext: filter(function(x) { return x % n === 0; }, nums)};
}, _.range(1, 12));

var powers = map(function(n) {
  return {ext: filter(function(x) {
    if (n === 1) return x === 1;
    // x is a power of n?
    var isPow = function(v) {
      return v === x ? true : (v > x ? false : isPow(v * n));
    };
    return isPow(1);
  }, nums)};
}, _.range(1, 12));

var evens = {ext: filter(function(x) { return x % 2 === 0; }, nums)};
var odds = {ext: filter(function(x) { return x % 2 === 1; }, nums)};

var ruleConcepts = multiples.concat(powers).concat([evens, odds]);
var ruleConcepts2 = filter(function(c) { return c.ext.length > 0; }, ruleConcepts);

// Interval concepts: a < b in [1,20]
var intervalPairs = _.flatten(map(function(a) {
  return map(function(b) {
    return {a: a, b: b};
  }, _.range(a + 1, maxNumber + 1));
}, _.range(1, maxNumber + 1)));

var intervalConcepts = map(function(p) {
  return {ext: filter(function(x) { return x >= p.a && x <= p.b; }, nums)};
}, intervalPairs);

var examples = [3, 6, 9];

var contains = function(ext, x) {
  return _.includes(ext, x);
};

var logLik = function(ext) {
  // each example uniform from ext
  var ok = reduce(function(e, acc) {
    return acc && contains(ext, e);
  }, true, examples);
  if (!ok) return -Infinity;
  return examples.length * (-Math.log(ext.length));
};

// prior: 50/50 mixture, equal within class
var nRule = ruleConcepts2.length;
var nInt = intervalConcepts.length;

var ruleEntries = map(function(c) {
  return {ext: c.ext, logPrior: Math.log(0.5) - Math.log(nRule)};
}, ruleConcepts2);

var intEntries = map(function(c) {
  return {ext: c.ext, logPrior: Math.log(0.5) - Math.log(nInt)};
}, intervalConcepts);

var allEntries = ruleEntries.concat(intEntries);

// posterior weights
var logWeights = map(function(e) {
  return e.logPrior + logLik(e.ext);
}, allEntries);

var maxLW = reduce(function(w, acc) { return Math.max(w, acc); }, -Infinity, logWeights);

var weights = map(function(lw) { return Math.exp(lw - maxLW); }, logWeights);
var totalW = sum(weights);

var normWeights = map(function(w) { return w / totalW; }, weights);

var probs = map(function(x) {
  var contributions = map(function(i) {
    var e = allEntries[i];
    return normWeights[i] * (contains(e.ext, x) ? 1 : 0);
  }, _.range(allEntries.length));
  return sum(contributions);
}, nums);

var ANSWER = probs;