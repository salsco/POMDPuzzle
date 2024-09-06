import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class Cubic:
  def __init__(self,tens,label):
    self.tens=np.array(tens) # In case it's not a numpy tensor
    self.label=label

  def sum_cubic(self,other_cubic):
    return Cubic(self.tens+other_cubic.tens,self.label)

  def to_matr_vec(self):
    dim = self.tens.shape[0]
    Q = np.zeros((dim, dim))
    a = np.zeros(dim)
    for i in range(dim):
      for j in range(dim):
        for k in range(dim):
          if i == j:
            a[k] += self.tens[i, j, k]
          Q[i, j] += self.tens[i, j, k]

    return a, Q


  def to_quadratic(self):
    a, Q = self.to_matr_vec()

    dim = len(a)-1
    q_vec=Q[-1, :dim]
    Q_tilde = Q[:dim, :dim]+2*np.matmul(a[:dim], q_vec)

    new_q_vec=(Q[-1,-1]*a[:dim]+2*a[-1]*q_vec)/2
    quadr_matrix = np.zeros((len(a),len(a)))
    quadr_matrix[:dim,:dim]=Q_tilde
    quadr_matrix[-1,:dim]=new_q_vec
    quadr_matrix[:dim,-1]=new_q_vec

    intersect=Q[-1,-1]*a[-1]


    
    return Quadratic(quadr_matrix,self.label)


class Quadratic:
  def __init__(self,quadr_matr,label):
    self.quadr_matr=np.array(quadr_matr) # In case it's not a matrix
    self.label=label

  @staticmethod
  def linear_regr_quadratic(f, D, label, n_samples=100):
    # Generiamo dei dati di esempio
    X = np.random.rand(n_samples, D)  # Punti casuali nell'iperspazio
    y = np.array([f(P) for P in X])  # Valori della funzione f per ogni punto

    # Creiamo le caratteristiche polinomiali di grado 2
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)


    # Applichiamo la regressione lineare
    model = LinearRegression()
    model.fit(X_poly, y)

    # Inizializziamo la matrice Q
    Q = np.zeros((D+1, D+1))

    # Popoliamo la matrice Q con i coefficienti appropriati
    # [1, linear_terms, a^2, ab,ac,...,b^2,bc,...]
    for i in range(D):
      Q[i, i] = model.coef_[1+D+(D-1-(i-1))*i+i]
      Q[i,-1]=model.coef_[1+i]/2
      Q[-1,i]=model.coef_[1+i]/2
      for j in range(i+1,D):
        Q[i,j]=model.coef_[1+D+(D-1-(i-1))*i+i+j]/2
        Q[j,i]=Q[i,j]


    # Aggiungiamo il termine noto alla matrice Q
    Q[-1, -1] = model.intercept_

    quadr_approx=Quadratic(Q,label)

    return quadr_approx

  def evaluate(self, point):
    completed_point=np.append(point,1)
    evaluation=np.dot(np.dot(completed_point[None], self.quadr_matr), completed_point)[0]
    return evaluation

  def sum_quadratic(self,quadratic):
    new_quadr_matr=self.quadr_matr+quadratic.quadr_matr

    return Quadratic(new_quadr_matr,self.label)

  def sum_to_linear(self,linear):
    Q=self.quadr_matr
    q_vec=Q[-1,:-1]
    a=linear.a

    new_Q=np.zeros((Q.shape[0],Q.shape[1]))
    new_Q[:-1,:-1]=Q[:-1,:-1]
    new_Q[-1,:-1]=q_vec+a[:-1]/2
    new_Q[:-1,-1]=q_vec+a[:-1]/2
    new_Q[-1,-1]=Q[-1,-1]+a[-1]
    return Quadratic(new_Q,self.label)

  def linearize_quadratic(self):
    # x^T Q x
    lin_vec=np.zeros((self.quadr_matr.shape[0]))
    lin_vec[:-1]=self.quadr_matr[:-1,-1].transpose()+self.quadr_matr[-1,:-1]
    lin_vec[-1]=self.quadr_matr[-1,-1]
    return Linear(lin_vec,self.label)

  def linear_transformation(self,matr):
    # x^T A^T Q A x
    new_quadr_matr=np.matmul(matr.transpose(),np.matmul(self.quadr_matr,matr))
    return Quadratic(new_quadr_matr,self.label)

  def multiply_to_linear(self,linear):
    dim = self.quadr_matr.shape[0]
    tens = np.zeros((dim, dim, dim))
    for i in range(dim):
      for j in range(dim):
        for k in range(dim):
          tens[i, j, k] = self.quadr_matr[i, j] * linear.a[k]

    return Cubic(tens,self.label)

  def scalar_multiply(self,scalar):
    new_quadr_matr=self.quadr_matr*scalar
    return Quadratic(new_quadr_matr,self.label)

  
  def quadraticform_string(self,decs=3):
    raw_Q=self.quadr_matr
    Q=np.trunc(raw_Q*10**decs)/(10**decs)
    d = Q.shape[0]
    variables = [f'x{i+1}' for i in range(d-1)]
    variables+=["*1"]
    terms = []

    # Termini quadratici
    for i in range(d):
        terms.append(f"{Q[i, i]}{variables[i]}^2")

    # Termini misti
    for i in range(d):
        for j in range(i + 1, d):
            terms.append(f"{2 * Q[i, j]}{variables[i]}{variables[j]}")


    return " + ".join(terms)



class Linear:
  # Linear in the form of z=[a1,a2,a3]^T*[x1,x2,1] (1 for the bias)
  def __init__(self,a,label):
    self.a=np.array(a) # In case it's not a numpy aray
    self.label=label

  def evaluate(self,point):
    return np.dot(self.a[:-1],point)+self.a[-1]

  def linear_transformation(self,l_transf):
    # l_transf is Cx, [C,d]
    # a^Tx; a^T(Cx)=a^T*C*x
    new_a=np.matmul(self.a.transpose(),l_transf["matr"])
    return Linear(new_a,self.label)

  def sum_to_linear(self,linear):
    new_a=self.a+linear.a
    return Linear(new_a,self.label)

  def __str__(self):
    return "Hyperplane "+str(self.a)+" of label "+str(self.label)


class QuadraticSet:
  def __init__(self,quadratics):
    self.quadratics=quadratics


  def sum_quadraticset(self,quadratic_set):
    new_quadratics=[]

    if(len(self.quadratics)!=len(quadratic_set.quadratics)):
      return None

    for quad_idx,quad in enumerate(self.quadratics):
      new_quadratics.append(quad.sum_quadratic(quadratic_set.quadratics[quad_idx]))

    return QuadraticSet(new_quadratics)

  def linearize_quadratics(self):
    linears=[]
    for quad in self.quadratics:
      linears.append(quad.linearize_quadratic())

    return LinearSet(linears)

  def maximum_evaluation(self,point):
    max_eval=None
    max_label=None
    for q_idx,quad in enumerate(self.quadratics):
      evaluation=quad.evaluate(point)
      if(q_idx==0):
        max_eval=evaluation
        max_label=quad.label
      else:
        if(evaluation>max_eval):
          max_eval=evaluation
          max_label=quad.label

    return max_eval,max_label

  def scalar_multiply(self,scalar):
    new_quadratics=[]
    for q in self.quadratics:
      new_quadratics.append(Quadratic(q.quadr_matr*scalar,q.label))

    return QuadraticSet(new_quadratics)


  def pretty_print(self):
    for quad in self.quadratics:
      print(quad.label)
      print(quad.quadr_matr)


class LinearSet:
  def __init__(self,linears):
    self.linears=linears      # Linears are given in the form z=ax+b, with a as numpy array and b as scalar

  def maximum_evaluation(self,point):
    max_eval=None
    max_label=None
    for lin_idx,lin in enumerate(self.linears):
      evaluation=lin.evaluate(point)
      if(lin_idx==0):
        max_eval=evaluation
        max_label=lin.label
      else:
        if(evaluation>max_eval):
          max_eval=evaluation
          max_label=lin.label

    return max_eval,max_label

  def reassign_all_labels(self,label):
    new_linears=[]
    for lin in self.linears:
      new_linears.append(Linear(lin.a,label))
    return LinearSet(new_linears)

  def merge_linearset(self,linearset):
    new_linears=self.linears.copy()
    for lin in linearset.linears:
      new_linears.append(lin)

    return LinearSet(new_linears)

  def scalar_multiply(self,scalar):
    new_linears=[]
    for lin in self.linears:
      new_linears.append(Linear(lin.a*scalar,lin.label))

    return LinearSet(new_linears)

  def sum_to_linearset(self,linearset):

    if(len(self.linears)!=len(linearset.linears)):
      return None

    new_linears=[]

    for lin_idx,lin in enumerate(self.linears):
      new_linears.append(lin.sum_to_linear(linearset.linears[lin_idx]))

    return LinearSet(new_linears)

  def multiply_to_linear(self,linear):
    quadratics=[]
    # z1=a^Tx
    # z2=c^Tx
    # z1*z2=a^Tx*c^Tx=x*a*c^T*x

    for lin in self.linears:
      quadr_matr=np.matmul(lin.a[None].transpose(),linear.a[None])
      quadratics.append(Quadratic(quadr_matr,lin.label))

    return QuadraticSet(quadratics)

  def multiply_to_linearset(self,linearset):
    if(len(self.linears)!=len(linearset.linears)):
      return None

    quadratics=[]
    for l_idx,lin in enumerate(self.linears):
      quadr_matr=np.matmul(lin.a[None].transpose(),linearset.linears[l_idx].a[None])
      quadratics.append(Quadratic(quadr_matr,lin.label))

    return QuadraticSet(quadratics)

  def linear_transformation(self,l_transf):
    new_linears=[]
    for lin in self.linears:
      new_linears.append(lin.linear_transformation(l_transf))

    return LinearSet(new_linears)

  def pretty_print(self):
    for lin in self.linears:
      print(lin.label)
      print(lin.a)
