var maxNumber = 20;
var allNums = _.range(1, maxNumber + 1);

// Rule-based concepts
var multiples = map(function(n) {
  return {name: 'multiples_' + n, set: filter(function(x) { return x % n === 0; }, allNums)};
}, _.range(1, 12));

var powers = map(function(n) {
  var ps = filter(function(x) {
    if (n === 1) return x === 1;
    // x is a power of n
    var check = function(v) { return v > x ? false : (v === x ? true : check(v * n)); };
    return check(1);
  }, allNums);
  return {name: 'powers_' + n, set: ps};
}, _.range(1, 12));

var evens = {name: 'evens', set: filter(function(x) { return x % 2 === 0; }, allNums)};
var odds = {name: 'odds', set: filter(function(x) { return x % 2 === 1; }, allNums)};

var ruleConcepts = multiples.concat(powers).concat([evens, odds]);

// Interval concepts
var intervalPairs = _.flatten(map(function(a) {
  return map(function(b) {
    return {a: a, b: b};
  }, _.range(a + 1, maxNumber + 1));
}, _.range(1, maxNumber)));

var intervalConcepts = map(function(p) {
  return {name: 'interval_' + p.a + '_' + p.b,
          set: _.range(p.a, p.b + 1)};
}, intervalPairs);

var observed = [3, 10];
var testQuery = 12;

var model = function() {
  var isRule = flip(0.5);
  var concept = isRule
    ? ruleConcepts[randomInteger(ruleConcepts.length)]
    : intervalConcepts[randomInteger(intervalConcepts.length)];

  var setArr = concept.set;
  // size principle likelihood
  map(function(obs) {
    condition(_.includes(setArr, obs));
    factor(-Math.log(setArr.length));
  }, observed);

  return {hypothesis: concept.name,
          testQueryResponse: _.includes(setArr, testQuery)};
};

var ANSWER = Infer({method: 'enumerate'}, model);