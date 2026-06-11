var maxNumber = 20;
var allInts = _.range(1, maxNumber + 1);

// Rule-based concepts
var multiplesRules = map(function(n) {
  return {name: 'multiples_' + n, ext: filter(function(x) { return x % n === 0; }, allInts)};
}, _.range(1, 12));

var powersRules = map(function(n) {
  return {name: 'powers_' + n, ext: _.uniq(filter(function(x) { return x >= 1; },
    map(function(e) { return Math.pow(n, e); }, _.range(0, 21))))};
}, _.range(1, 12));

var evensRule = [{name: 'evens', ext: filter(function(x) { return x % 2 === 0; }, allInts)}];
var oddsRule = [{name: 'odds', ext: filter(function(x) { return x % 2 === 1; }, allInts)}];

var ruleConcepts = multiplesRules.concat(powersRules).concat(evensRule).concat(oddsRule);
var validRuleConcepts = filter(function(c) { return c.ext.length > 0; }, ruleConcepts);

// Interval concepts: 1 <= a < b <= 20
var intervalPairs = _.flatten(map(function(a) {
  return map(function(b) {
    return {a: a, b: b};
  }, _.range(a + 1, maxNumber + 1));
}, _.range(1, maxNumber)));

var intervalConcepts = map(function(p) {
  return {name: 'interval_' + p.a + '_' + p.b,
          ext: _.range(p.a, p.b + 1)};
}, intervalPairs);

var observed = [3, 10];
var testQuery = 12;

var model = function() {
  var isRule = flip(0.5);
  var concept = isRule
    ? validRuleConcepts[randomInteger(validRuleConcepts.length)]
    : intervalConcepts[randomInteger(intervalConcepts.length)];

  var ext = concept.ext;
  // size principle likelihood
  map(function(obs) {
    observe(Categorical({vs: ext, ps: map(function(x) { return 1; }, ext)}), obs);
  }, observed);

  var inSet = _.includes(ext, testQuery);
  return {hypothesis: concept.name, testQueryResponse: inSet};
};

var ANSWER = Infer({method: 'enumerate'}, model);