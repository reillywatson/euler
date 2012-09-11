import math
from collections import deque
import itertools
from fractions import Fraction

def prime_factors_gen(n):
	yield 1
	i = 2
	limit = n**0.5
	while i <= limit:
		if n % i == 0:
			yield i
			n = n /i
			limit = n**0.5
		else:
			i += 1
	if n > 1:
		yield n

def prime_factors(x):
	factors = []
	p = [1] + primes(x**0.5)
	return [a for a in p if x % a == 0]

def lowest_common_multiple(seq):
	largest = max(seq)
	i = largest
	while True:
		if [x for x in seq if i % x == 0] == seq:
			return i
		i = i + largest

def sum_squares(seq):
	return sum([math.pow(x,2) for x in seq])

def square_sum(seq):
	return sum(seq) ** 2

def has_divisor(x, seq):
	for a in seq:
		if a > math.ceil(math.sqrt(x)):
			return False
		if x % a == 0:
			return True
	return False

# gets first n primes
def first_n_primes(n):
	x = [2]
	while len(x) < n:
		i = x[-1]
		while True:
			i = i + 1
			if not has_divisor(i, x):
				x.append(i)
				break
	return x

# gets primes smaller than or equal to n
# for problem 10
def primes(n): 
	if n==2: return [2]
	elif n<2: return []
	s=range(3,n+1,2)
	mroot = n ** 0.5
	half=(n+1)/2-1
	i=0
	m=3
	while m <= mroot:
		if s[i]:
			j=(m*m-3)/2
			s[j]=0
			while j<half:
				s[j]=0
				j+=m
		i=i+1
		m=2*i+3
	return [2]+[x for x in s if x]

def gen_primes():
	D = {}  
	q = 2  
	while True:
		if q not in D:
			yield q        
			D[q * q] = [q]
		else:
			for p in D[q]:
				D.setdefault(p + q, []).append(p)
			del D[q]
		q += 1

def is_prime(x):
	if x < 0:
		return False
	for i in range(2, int(math.sqrt(x) + 1)):
		if x % i == 0:
			return False
	return True

def product(seq):
	return reduce(lambda x,y: x * y, seq)

# problem 5
def smallest_divisible_by_all(seq):
	i = 1
	while True:
		divisible = True
		for a in seq:
			if i % a > 0:
				divisible = False
				break
		if divisible:
			return i
		i = i + 1

# problem 6
def prob6():
	return sum_squares(range(1,101)) - square_sum(range(1,101))

# problem 8
def greatest_product(s):
	max = 0
	for i in range(len(s)-5):
		p = product([int(a) for a in s[i:i+5]])
		if p > max:
			max = p
	return max
	
# problem 9
# gets all Pythagorean triplets below n
def pythagorean_triplets(n):
	trips = []
	for i in range(1,n-2):
		for j in range(i,n-1):
			sum = math.pow(i, 2) + math.pow(j, 2)
			if math.sqrt(sum) == int(math.sqrt(sum)):
				trips.append([i, j, int(math.sqrt(sum))])
	return trips

# 4-tuples of adjacent entries in a grid (horizontal, vertical, and diagonal)
# for problem 11
def adjacent_4tuples(grid):
	tuples = []
	width = len(grid[0])
	height = len(grid)
	for i in range(width):
		for j in range(height):
			if i + 3 < height:
				tuples.append([grid[i][j],grid[i+1][j],grid[i+2][j],grid[i+3][j]])
			if j + 3 < width:
				tuples.append([grid[i][j],grid[i][j+1],grid[i][j+2],grid[i][j+3]])
			if i + 3 < height and j + 3 < width:
				tuples.append([grid[i][j],grid[i+1][j+1],grid[i+2][j+2],grid[i+3][j+3]])
			if i + 3 < height and j >= 3:
				tuples.append([grid[i][j],grid[i+1][j-1],grid[i+2][j-2],grid[i+3][j-3]])
	return tuples

def to_ints(seq):
	ints = []
	for a in seq:
		if isinstance(a, list):
			ints.append(to_ints(a))
		else:
			ints.append(int(a))
	return ints

def divisors(x):
	divisors = set()
	for i in range(1, int(math.ceil(math.sqrt(x))) + 1):
		if x % i == 0:
			divisors.add(i)
			divisors.add(x / i)
	return sorted(list(divisors))

# problem 11
def triangle_number_gen():
	sum = 0
	i = 0
	while True:
		i = i + 1
		sum = sum + i
		yield sum


def problem12():
	gen = triangle_number_gen()
	cur = 1
	while len(divisors(cur)) < 500:
		cur = gen.next()
		print cur
	return cur

# problem 14
gKnownCollatz={}	
def collatz(n):
	if gKnownCollatz.has_key(n):
		return gKnownCollatz[n]
	l = []
	if n == 1:
		l = [1]
	elif n % 2 == 0:
		l = [n]
		l.extend(collatz(n / 2))
	else:
		l = [n]
		l.extend(collatz(n*3+1))
	gKnownCollatz[n] = l
	return l

# problem 15
# without this global this is crazy slow, with it it takes < 1s
gKnownGridRoutes = {} 
def gridroutes(x,y):
	if x == 1 or y == 1:
		return 1
	if gKnownGridRoutes.has_key((x,y)):
		return gKnownGridRoutes[(x,y)]
	ans = gridroutes(x-1, y) + gridroutes(x, y-1)
	gKnownGridRoutes[(x,y)] = ans
	return ans

def prob15():
	return gridroutes(21,21)

def prob16():
	return sum([int(a) for a in list(str(2 ** 1000))])

# problem 17
def numtowordbad(n, addAnd=False):
	if n < 20:
		result = teenword(n)
		if result != '' and addAnd:
			result = 'and ' + result
		return result
	elif n < 100:
		result = tensword(n) + numtowordbad(n%10, False)
		if result != '' and addAnd:
			result = 'and ' + result
		return result
	elif n < 1000:
		remainder = numtowordbad(n % 100, True)
		if remainder != '':
			remainder = ' ' + remainder
		return teenword(n/100)+' hundred' + remainder
	elif n == 1000:
		return 'one thousand'
	return ''

# better implementation, but excludes "and"s, American-style, so Euler doesn't like it
def numtoword(n):
	if n < 0:
		return 'negative ' + numtoword(-n)
	if n < 20:
		return teenword(n)
	elif n < 100:
		return tensword(n) + numtoword(n%10)	
	elif n < 1000:
		remainder = numtoword(n % 100)
		if remainder != '':
			remainder = ' ' + remainder
		return teenword(n/100)+' hundred' + remainder
	else:
		ids = ['thousand','million','billion','trillion', 'quadrillion','quintillion','sextillion','septillion','octillion','nonillion',
		'decillion','undecillion','duodecillion','tredecillion','quattuordecillion','quindecillion','sexdecillion','septendecillion',
		'octodecillion','novemdecillion','vigintillion']
		log = int(math.log(n, 1000))
		if log > len(ids):
			raise ValueError('this number is waaay too big!')
		pow = int(math.pow(10, log * 3))
		remainder = numtoword(n % pow)
		if remainder != '':
			remainder = ' ' + remainder
		return numtoword(n/pow) + ' ' + ids[log - 1] + remainder

def tensword(n):
	x = ''
	if n < 30:
		x = 'twenty'
	elif n < 40:
		x = 'thirty'
	elif n < 50:
		x = 'forty'
	elif n < 60:
		x = 'fifty'
	elif n < 70:
		x = 'sixty'
	elif n < 80:
		x = 'seventy'
	elif n < 90:
		x = 'eighty'
	elif n < 100:
		x = 'ninety'
	if x != '' and n % 10 > 0:
		x = x + '-'
	return x

def teenword(n):
	nums = ['one','two','three','four','five','six','seven','eight','nine','ten',
	'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
	if n > 0 and n < 20:
		return nums[n-1]
	return ''

def prob17():
	x = ''
	for a in range(1,1001):
		x = x + numtowordbad(a)
	x = x.replace(' ', '').replace('-','')
	return len(x)

# problem 18
#data = [[75], [95, 64], [17, 47, 82], [18, 35, 87, 10], [20, 4, 82, 47, 65], [19, 1, 23, 75, 3, 34], [88, 2, 77, 73, 7, 63, 67], [99, 65, 4, 28, 6, 16, 70, 92], [41, 41, 26, 56, 83, 40, 80, 70, 33], [41, 48, 72, 33, 47, 32, 37, 16, 94, 29], [53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14], [70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57], [91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48], [63, 66, 4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31], [4, 62, 98, 27, 23, 9, 70, 98, 73, 93, 38, 53, 60, 4, 23]]
def triangle_sums(data):
	sums = data[-1]
	for i in range(len(data) - 2, -1, -1):
		print sums
		rowsums = []
		rowdata = data[i]
		for j in range(0, len(rowdata)):
			rowsums.append(rowdata[j]+max(sums[j],sums[j+1]))
		sums = rowsums
	return sums[0]

def days_in_month(month,year):
	if month in [1, 3, 5, 7, 8, 10, 12]:
		return 31
	elif month == 2:
		if year % 4 == 0 and not (year % 100 == 0 and year % 400 > 0):
			return 29
		return 28
	return 30

# problem 19
def prob19():
	year = 1900
	month = 1
	day = 1
	weekday = 1
	sundays = 0
	while year < 2001:
		day = day + 1
		if day > days_in_month(month, year):
			day = 1
			month = month + 1
		if month > 12:
			month = 1
			year = year + 1
		weekday = weekday + 1
		if weekday > 7:
			weekday = 1
		if weekday == 7 and day == 1 and year >= 1901 and year < 2001:
			sundays = sundays + 1
	return sundays

# problem 20
def prob20():
	return sumdigits(math.factorial(100))

# problem 21
def proper_divisors(n):
	d = divisors(n)
	d.remove(n)
	return d

def amicable(n):
	amies = set()
	for a in range(1,n):
		divs = sum(proper_divisors(a))
		if divs > 0 and sum(proper_divisors(divs)) == a and a != divs:
			amies.add(a)
			amies.add(divs)
	return sum(amies)

# problem 22
def prob22():
	names = sorted(open('/home/reilly/names.txt').read().lower().split(','))
	lineNo = 0
	score = 0
	for name in names:
		lineNo = lineNo + 1
		letters = '"abcdefghijklmnopqrstuvwxyz'
		name = name.lower()
		print name, score
		s = sum([letters.index(a) for a in name])
		score = score + (s * lineNo)
	return score

# problem 23
def abundant(n):
	x = []
	for i in range(1,n):
		if sum(proper_divisors(i)) > i:
			x.append(i)
	return x

def noabundantsum():
	abundants = abundant(28123)
	sums = set()
	for a in abundants:
		for b in abundants:
			sums.add(a+b)
	return sum([x for x in range(1,28123) if x not in sums])

def prob24():
	x = itertools.permutations([1,2,3,4,5,6,7,8,9,0])
	l = sorted([a for a in x])
	return ''.join([str[a] for a in l[999999]])

# problem 24: n2 - 61n + 971 is best (70 primes)
def quadraticprime():
	p = [1]
	p.extend(primes(1000))
	for a in p:
		for b in p:
			max = 0
			for i in range(a*b):
				result = math.pow(i,2) + a*i + b
				if not is_prime(result):
					break
				max = i
			if max > 30:
				print 'try this one: n2 + '+str(a)+'n + '+str(b)+', got ' + str(max) + ' results'
			max = 0
			for i in range(a*b):
				result = math.pow(i,2) - a*i + b
				if not is_prime(result):
					break
				max = i
			if max > 30:
				print 'try this one: n2 - '+str(a)+'n + '+str(b)+', got ' + str(max) + ' results'
			max = 0
			for i in range(a*b):
				result = math.pow(i,2) - a*i - b
				if not is_prime(result):
					break
				max = i
			if max > 30:
				print 'try this one: n2 - '+str(a)+'n - '+str(b)+', got ' + str(max) + ' results'
			max = 0
			for i in range(a*b):
				result = math.pow(i,2) + a*i - b
				if not is_prime(result):
					break
				max = i
			if max > 30:
				print 'try this one: n2 + '+str(a)+'n - '+str(b)+', got ' + str(max) + ' results'

	return p

# problem 25
# returns all the Fibonacci numbers smaller than n
def fibs(n):
	f = [1, 1]
	while n > f[-1]:
		f.append(f[-1]+f[-2])
	f.pop()
	return f

# returns the Nth Fibonacci number
def fib(n):
	a = 1
	b = 0
	i = 0
	while i < (n - 1):
		tmp = a
		a = a + b
		b = tmp
		i = i + 1
	return a


# problem 26
def longestreciprocalcycle():
	longest = []
	longi = 0
	for i in range(1, 1000):
		qr = []
		r = 1
		q=1
		print 'testing %d' % i
		while r != 0:
			r = q % i
			q = q / i
			if [q,r] in qr:
				break
			qr.append([q,r])
			q = r * 10
		if len(qr) > len(longest):
			longi = i
			longest = qr
		print 'length: %d' % len(qr)
	print 'longest: %d' % longi
	print 'length: %d' %len(longest)

# problem 27
# [-61, 971, 71]
def prob27():
	p = primes(5000)
	max = 0
	maxresult=[]
	for a in range(-1000,1000):
		print a
		minb = -1000
		maxb = 1000
		for b in range(minb,maxb):
			numprimes = 0
			for x in range(0,math.fabs(b)):
				if math.pow(x,2) + (a*x) + b in p:
					numprimes = numprimes + 1
				else:
					break
			if numprimes > max:
				max=numprimes
				maxresult = [a,b,numprimes]
				print 'score'
				print maxresult
	print maxresult

# problem 28: sum is 669171001
def spiraldiagonals(n):
	s = 1
	skip = 2
	c = 1
	print 1
	for i in range(int(n/2)):
		for j in range(4):
			c = c + skip
			print c
			s = s + c
		skip += 2
	return s

#problem 29: (9183 uniques)
def uniqexponents():
	ans = set()
	for a in range(2, 101):
		for b in range(2, 101):
			ans.add(math.pow(a,b))
	return ans
	
# problem 30 (sum is 443839)
def fifthpowersum():
	maxtest = 354295 # 9^5 * 6 + 1
	nums = []
	for i in range(2,maxtest):
		print i
		s = sum([math.pow(int(a),5) for a in str(i)])
		if s == i:
			nums.append(i)
	return nums

# problem 31 (73682 ways)
def knapsack(values, total):
	results = []
	tries = 0
	for pence in range(201):
		for tup in range(0,201 - pence,2):
			for nick in range(0,201 -pence-tup,5):
				for dime in range(0,201-pence-tup-nick,10):
					for twent in range(0,201-pence-tup-nick-dime,20):
						for fif in range(0,201-pence-tup-nick-dime-twent, 50):
							for pound in range(0,201-pence-tup-nick-dime-twent-fif,100):
								for two in range(0,201-pence-tup-nick-dime-twent-fif-pound, 200):
									tries = tries + 1
									if tries % 100 == 0:
										print [pence, tup, nick, dime, twent, fif, pound, two]
									if pence + (tup) + (nick)+(dime)+(twent)+(fif)+(pound)+(two) == 200:
										results.append([pence, tup, nick, dime, twent, fif, pound, two])
	return results


# problem 32
def is_pandigital(x, n=9):
	check = '123456789'[:n]
	for a in check:
		if not a in x:
			return False
	return len(x) == n

def prob32():
	res = []
	for cand in range(1, 10000):
		print cand
		for mul in range(1,1000):
			prod = cand * mul
			if is_pandigital(str(cand) + str(mul) + str(prod)):
				res.append([cand,mul,prod])
	return res

# problem 33 (100)
def prob33():
	fracs = []
	for a in range(10,100):
		for b in range(a + 1, 100):
			print a,b
			stra = str(a)
			strb = str(b)
			if len(stra) < 2 or len(strb) < 2:
				continue
			if stra[1] == '0' or strb[1] == '0':
				continue
			if stra[0] == strb[1]:
				if float(stra[1]) / float(strb[0]) == float(stra) / float(strb):
					fracs.append([a,b])
			if stra[1] == strb[0]:
				if float(stra[0]) / float(strb[1]) == float(stra) / float(strb):
					fracs.append([a,b])
	return fracs

#problem 34 [145, 40585]
def prob34():
	nums = []
	factorials = [math.factorial(a) for a in range(10)]
	for i in range(3, 100000000):
		if i % 10000 == 0:
			print i
		s = str(i)
		total = sum([factorials[int(a)] for a in s])
		if i == total:
			nums.append(i)
	return nums
	
# problem 35
def circular_primes(n):
	p = primes(n)
	circ = []
	for x in p:
		print x
		good = True
		a = deque([i for i in str(x)])
		for i in range(len(a)):
			a.append(a.popleft())
			if int(''.join(a)) not in p:
				good = False
				break
		if good:
			circ.append(x)
	return circ
	
# problem 36
def is_palindrome(s):
	for i in range(int(math.floor(len(s)/2.0))):
		if s[i] != s[-i-1]:
			return False
	return True
				
def prob36():
	pals = []
	for i in range(1,1000000):
		if is_palindrome(str(i)):
			if is_palindrome(bin(i).replace('0b','')):
				pals.append(i)
	return sum(pals)
	
# problem 37
# [23, 37, 53, 73, 313, 317, 373, 797, 3137, 3797, 739397]
def prob37(n):
	print 'generating'
	p = primes(n)
	onedigit = primes(10)
	twodigit = primes(100)
	realprimes = []
	for a in p:
		astr = str(a)
		if int(astr[:2]) in twodigit and int(astr[0]) in onedigit:
			realprimes.append(a)
	print 'generate'
	res= []
	for x in realprimes:
		if x > 10 and cantrunc(x, p):
			res.append(x)
	return res

def cantrunc(x, p):
	a = deque([i for i in str(x)])
	while len(a) > 0:
		if not int(''.join(a)) in p:
			return False
		a.pop()
	a = deque([i for i in str(x)])
	while len(a) > 0:
		if not int(''.join(a)) in p:
			return False
		a.popleft()
	return True

# problem 38
def prob38():
	valids = [i for i in range(99999) if len(set(str(i))) == len(str(i))]
	print valids
	for i in valids:
		s = ''
		j = 1
		while len(s) < 9:
			s = s + str(i * j)
			j = j + 1
		if is_pandigital(s):
			print i, s


# problem 39
def maxperim(n):
	trips = [x for x in pythagorean_triplets(n) if sum(x) <= n]
	counts = {}
	for a in trips:
		if sum(a) in counts:
			counts[sum(a)] = counts[sum(a)]+1
		else:
			counts[sum(a)] = 1
	return counts

def prob39():
	x = maxperim(1000)
	largestkey = 0
	largestval = 0
	for a in x:
		if x[a] > largestval:
			largestkey = a
			largestval = x[a]
	print 'key: ' + str(largestkey) + ' val: ' + str(largestval)

# problem 40
def prob40():
	s = ''
	for i in range(1, 1000000):
		s = s + str(i)
	return int(s[0]) * int(s[9]) * int(s[99]) * int(s[999]) * int(s[9999]) * int(s[99999]) * int(s[999999])

# problem 42
def irrational_fraction_digit(n):
	x = '0'
	i = 1
	while len(x) <= n:
		x = x + str(i)
		i = i + 1
	return int(x[n])
	
def prob42():
	return irrational_fraction_digit(1)* \
	irrational_fraction_digit(10)* \
	irrational_fraction_digit(100)* \
	irrational_fraction_digit(1000)* \
	irrational_fraction_digit(10000)* \
	irrational_fraction_digit(100000)* \
	irrational_fraction_digit(1000000)

# problem 41
def primepandigitals():
	print 'generating'
	p = primes(10000000)
	print 'generated'
	pans = []
	for x in p:
		if is_pandigital(str(x), len(str(x))):
			pans.append(x)
	return pans

# problem 42
def trianglewords():
	letters = '"abcdefghijklmnopqrstuvwxyz'
	list = open('/home/reilly/Downloads/words.txt').read().lower().split(',')
	gen = triangle_number_gen()
	nums = [0]
	words = []
	for word in list:
		print word
		s = sum([letters.index(a) for a in word])
		while s > max(nums):
			nums.append(gen.next())
		if s in nums:
			words.append(word)
	return words

# problem 43
def prob43():
	nums = []
	for i in itertools.permutations([1,2,3,4,5,6,7,8,9,0]):
		s = ''.join([str(x) for x in i])
		if int(s[1:4]) % 2 == 0 \
		and int(s[2:5]) % 3 == 0 \
		and int(s[3:6]) % 5 == 0 \
		and int(s[4:7]) % 7 == 0 \
		and int(s[5:8]) % 11 == 0 \
		and int(s[6:9]) % 13 == 0 \
		and int(s[7:10]) % 17 == 0:
			nums.append(int(s))
	return nums

# problem 44 (5482660)
def sumdifpent():
	gen = pentagonal_number_gen()
	pents = []
	matches = []
	for i in range(5000):
		pents.append(gen.next())
	for i in range(len(pents)):
		a = pents[i]
		print i, a
		for j in range(i, len(pents)):
			b = pents[j]
			if a + b in pents and abs(a - b) in pents:
				matches.append([a,b])
	return matches

# problem 45
def pentagonal_number_gen():
	n = 0
	while True:
		n = n + 1
		yield n * (3*n-1) / 2

def hexagonal_number_gen():
	n = 0
	while True:
		n = n + 1
		yield n*(2*n-1)

def prob45():
	gens = [triangle_number_gen(), pentagonal_number_gen(), hexagonal_number_gen()]
	vals = [40756,1,1]
	while vals[0] != vals[1] or vals[1] != vals[2]:
		i = vals.index(min(vals))
		vals[i] = gens[i].next()
	return vals

# problem 46
def square_gen():
	n = 0
	while True:
		n = n + 1
		yield n * n

def goldbachoddcomposite():
	squares = [0]
	gen = square_gen()
	p = primes(100000)
	i = 7
	while True:
		i = i + 2
		found = False
		if not i in p:
			while i > max(squares):
				squares.append(gen.next())
			for j in squares:
				x = 2 * j
				if x < i:
					r = i - x
					if r in p:
						found = True
						break
			if not found:
				return i


#problem 47
def prob47(n):
	i = 1
	numconsecutive = 0
	while True:
		print i
		factors = set(prime_factors_gen(i))
		if len(factors) == n + 1:
			numconsecutive = numconsecutive + 1
		else:
			numconsecutive = 0
		if numconsecutive > 3:
			print i, factors
		if numconsecutive == n:
			return i - n + 1
		i = i + 1

# problem 48
def prob48():
	x = 0
	for i in range(1,1001):
		print i
		x = x + (i**i)
	return x

#problem 49
def prob49():
	p = [a for a in primes(10000) if a > 1000]
	trips = []
	for i in p:
		print i
		bigger = [a for a in p if a > i]
		for j in bigger:
			diff = j - i
			otherone = [a for a in p if a - (diff * 2) == i]
			if len(otherone) > 0:
				if set(str(i)) == set(str(j)) == set(str(otherone[0])):
					trips.append([i,j,otherone[0]])
	return trips

# problem 50 (997651, 543 consecutive primes from 7 onward)
def sequentialprimes(n):
	p = primes(n)
	longest = []
	print 'num primes: %d' % len(p)
	for i in range(10):
#		if i % 1000 == 0:
		print p[i]
		s = 0
		for j in range(len(p) - i):
			s = s + p[i+j]
			if s > n:
				break
			if is_prime(s):
				if j > len(longest):
					longest = p[i:i+j+1]
	return longest

def prob50():
	return sequentialprimes(1000000)

# problem 51 (121313)
def digitcounts(x):
	counts = [0,0,0,0,0,0,0,0,0,0]
	for a in str(x):
		counts[int(a)] = counts[int(a)] + 1
	return counts

def prob51(n):
	allp = primes(n)
	fams = []
	p = []
	for test in allp:
		if max(digitcounts(test)) > 2:
			p.append(test)
	for a in range(len(p)):
		x = p[a]
		xstr = str(x)
		counts = digitcounts(x)
		testspots = []
		for c in range(10):
			if counts[c] > 2:
				fams.append(xstr.replace(str(c), '*'))
	for a in fams:
		n = 0
		for i in range(10):
			if int(a.replace('*', str(i))) in p:
				n = n + 1
				if n > 7:
					print a, n

def sort_by_most_instances(seq):
	d = {}
	for i in seq:
		if i in d:
			d[i] = d[i] + 1
		else:
			d[i] = 1
	return sorted(d.keys(), key=lambda x:d[x])

# problem 52
def prob52():
	i = 1
	while True:
		if set(str(i*2)) == set(str(i*3)) == set(str(i*4)) == set(str(i*5)) == set(str(i*6)):
			return i
		i = i + 1

# problem 53
def NchooseR(n,r):
	return math.factorial(n) / (math.factorial(r) * math.factorial(n-r))

def prob53():
	vals = 0
	for n in range(1, 101):
		for r in range(1, n):
			if NchooseR(n,r) > 1000000:
				vals = vals + 1
	return vals

# problem 54 (376)
# concept: rank hands by assigning them values.
# Straight flush: 10000000xxxxx
# 4 of a kind:     1000000xxxxx
# Full house:       100000xxxxx
# Flush:             10000xxxxx
# Straight:           1000xxxxx
# 3 of a kind:         100xxxxx
# 2 pair:               10xxxxx
# 1 pair:                1xxxxx
# high card:              xxxxx
# xxxxx represents your cards' values, most important cards first
def cardval(card):
	if card == 'T':
		return 10
	if card == 'J':
		return 11
	elif card == 'Q':
		return 12
	elif card == 'K':
		return 13
	elif card == 'A':
		return 14
	return int(card)

def biggest_set(vals):
	return max(counts(vals).values())

def counts(seq):
	counts = {}
	for i in seq:
		if i in counts:
			counts[i] = counts[i] + 1
		else:
			counts[i] = 1
	return counts	

def pair_count(vals):
	return len([x for x in counts(vals).values() if x >= 2])

def is_straight(vals):
	if vals == [2,3,4,5,14]:
		return True
	if biggest_set(vals) > 1:
		return False
	return max(vals) - min(vals) == 4

def pairwise_sort(vals):
	result = []
	c = counts(vals)
	s = sort_by_most_instances(vals)
	for i in s:
		for a in range(c[i]):
			result.append(i)
	return result

def pokerlisttostring(vals):
	s = ''
	for a in vals:
		if a == 14 and is_straight(vals) and min(vals) == 2:
			a = 1
		s = s + hex(a)[2:]
	return s	

def pokerhandvalue(hand):
	suits = [s[1] for s in hand]
	vals = sorted([cardval(s[0]) for s in hand], reverse=True)
	if len(set(suits)) == 1:
		if is_straight(vals):
			return '10000000' + pokerlisttostring(vals)
		else:
			return '10000' + pokerlisttostring(vals)
	elif is_straight(vals):
		return '1000' + pokerlisttostring(vals)
	elif biggest_set(vals) == 4:
		return '1000000' + pokerlisttostring(list(reversed(pairwise_sort(vals))))
	elif biggest_set(vals) == 3:
		if pair_count(vals) == 2:
			return '100000' + pokerlisttostring(list(reversed(pairwise_sort(vals))))
		else:
			return '100' + pokerlisttostring(list(reversed(pairwise_sort(vals))))
	elif biggest_set(vals) == 2:
		if pair_count(vals) == 2:
			return '10' + pokerlisttostring(list(reversed(pairwise_sort(vals))))
		else:
			return '1' + pokerlisttostring(list(reversed(pairwise_sort(vals))))
	return pokerlisttostring(vals)

def prob54():
	p1wins = 0
	for line in open('/home/reilly/Downloads/poker.txt'):
		line = line.lstrip().rstrip()		
		cards = line.split(' ')
		hand1 = cards[:5]
		hand2 = cards[5:]
		val1 = int(pokerhandvalue(hand1),16)
		val2 = int(pokerhandvalue(hand2), 16)
		if val1 > val2:
			p1wins = p1wins + 1
		print hand1, val1
		print hand2, val2
		print val1 > val2
	return p1wins

# problem 55
def reverse(s):
	return s[::-1]

def lychrel(n):
	vals = []
	for i in range(n):
		a = i + int(reverse(str(i)))
		islychrel=True
		for j in range(50):
			if is_palindrome(str(a)):
				islychrel = False
				break
			a = a + int(reverse(str(a)))
		if islychrel:
			vals.append(i)
	return vals

def prob55():
	return len(lychrel(10000))

# problem 56
def sumdigits(x):
	return sum([int(a) for a in str(x)])

def prob56(): # this looks like it should be 978 (88^99)...why doesn't the site believe me?
	max = 0
	for a in range(1,100):
		for b in range(1,100):
			s = sumdigits(a ** b)
			if s > max:
				print a, b, s
				max = s
	return max

gfracs = {1 : Fraction(2)}
def frac(iters):
	if iters in gfracs:
		return gfracs[iters]
	result = Fraction(2) + Fraction(1) / frac(iters-1)
	gfracs[iters] = result
	return result

# problem 57
def prob57():
	result = 0
	for i in range(1, 1001):
		n = Fraction(1) + Fraction(1) / frac(i)
		if len(str(n.numerator)) > len(str(n.denominator)):
			result = result + 1
	return result

# problem 58 (26241)
def prob58():
	skip = 2
	c = 1
	vals = [1]
	primenos = []
	sidelength = 1
	while True:
		for j in range(4):
			c = c + skip
			vals.append(c)
			if is_prime(c):
				primenos.append(c)
		skip += 2
		ratio = float(len(primenos)) / len(vals)
		print ratio
		sidelength += 2
		if ratio < 0.1:
			return sidelength
	return None

# problem 62
def is_permutation(a,b):
	return sorted(a) == sorted(b)

def countof(seq, x):
	return len([a for a in seq if a == x])

def prob62(n):
	i = 1
	cubes = []
	sortedcubes = []
	while True:
		i = i + 1
		print i
		cube = str(i ** 3)
		cubes.append(cube)
		sortedcube = sorted(cube)
		sortedcubes.append(sortedcube)
		if countof(sortedcubes, sortedcube) == n:
			yield min([int(x) for x in cubes if sorted(x) == sorted(cube)])

def prob67():
	lines = []
	for a in open("/home/reilly/Downloads/triangle.txt"):
		lines.append([int(x) for x in a.split(' ')])
	return triangle_sums(lines)

# problem 79 (73162890)
def inorder(a, b):
	lastindex = 0
	for char in b:
		index = a.find(char, lastindex)
		if index == -1:
			return False
		lastindex = index + 1
	return True

def allinorder(a, set):
	for b in set:
		if not inorder(a,b):
			return False
	return True

def prob79():
	pwds = open('/home/reilly/Downloads/keylog.txt').read().split('\r\n')[:-1]
	digits = sorted(set(''.join(pwds)))
	midlength = 1
	while True:
		combinations = itertools.product(digits, repeat=midlength)
		for c in combinations:
			print ''.join(c)
			if allinorder(''.join(c), pwds):
				return ''.join(c)
		midlength = midlength + 1


# problem 85
def num_rectangles(width, height):
	numrects = 0
	for x in range(1, width+1):
		for y in range(1, height+1):
			numrects = numrects + (width - x + 1) * (height - y + 1)
	return numrects

# problem 97 (8739992577)
def prob97():
	t = 1
	for i in range(1, 7830458):
		t = (t * 2) % 100000000000
	return 28433 * t + 1

def prob201(set, size):
	for i in range(2, 3):
		comb_gen = itertools.combinations(set, i)
		for i in comb_gen:
			if sum(i) in set:
				set.remove(sum(i))
	return set


# new idea: this sequence grows on the left, the right hand side is stable.
# Since we're looking for the position when the first instance appears,
# we just need to calculate the fibonacci sequence to get the string length,
# rather than the actual string, then get the rightmost n digits.
# reduces the search space to log(space)

# idea: solve the problem for A=a, B=b.
# reduces the search space by a factor of 100
# problem 230
# generates the first n fib abbreviations
def fib2(a,b,n):
	x = a
	y = ''
	if n == 1:
		return a
	if n == 2:
		return b
	for i in range(1,n):
		print i, len(x)
		tmp = x
		x = y + x
		if y == '':
			y = b
		else:
			y = tmp
	x = x.replace('a','x')
	x = x.replace('b','a')
	x = x.replace('x','b')
	return x

def abbrevfib(a,b,n):
	if n == 1:
		return a
	if n == 2:
		return b
	ret = ''
	x = a
	y = b
	for i in range(n):
		tmp = x
		x = y + tmp
		y = tmp
	return x


def fibstr(a, b, n):
	abbrevs = abbrevfibs('a','b',500)
	print n
	xlist = [a,b]
	xstr = ''
	while len(xstr) < n:
		xlist.append(xlist[-2]+xlist[-1])
		xstr = xlist[-1]
	return xstr

def prob230():
	a = '1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'
	b = '8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196'
	sum = 0
	for n in range(18):
		ans = int(int(math.pow(10, n)) * fibstr(a, b, (127 + 19*n)*int(math.pow(7,n))))
		sum = sum + ans
	return sum


# problem 308
def nextieratation_gen(state):
	program = [(17,91),(78,85),(19,51),(23,38),(29,33),(77,29),(95,23),(77,19),(1,17),(11,13),(13,11),(15,2),(1,7),(55,1)]
	for a in program:
		if a[0]*state % a[1]:
			yield state * a[0] / a[1]

def prob308():
	gen = nextiteration_gen(2)

def gen_pi():
	i = 3
	pi = 1
	a = -1
	while True:
		yield pi * 4
		pi = pi + (1.0 / (i*a))
		i = i + 2
		a = -a

def gen_pi2():
	i = 1
	n = 0
	while True:
		n = n + (1 / i ** 2.0)
		yield (n*6)**.5
		i = i + 1
