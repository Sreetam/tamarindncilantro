# tamarindncilantro

1. Importing the library
	from transmission_graph import regions as tg

2. Creating a new transmission graph object: 
	
	g= tg.transmission_graph()

This will create an arbitrary graph with 9 nodes named 'A' to 'I' at arbitrary locations.


3. Adding a new node
	
	#Call the function on the object:-
	
	g.add_node(node_name, x, y, t1, r1, h1)
	
PARAMETERS:-
	node_name - name of the city	(string)
	x- x coordinate of city		(float)
	y- y coordinate of city		(float)
	t1- temperature of the city	(float)
	r1- rainfall of the city	(float)
	h1- humidity of the city	(float)

4. Removing a node

	g.remove_node(node_name)	(string)

5. Modifying the edge weight between two nodes:-

	g.change_weight(r1, r2, wt)

PARAMETERS:-
	r1- region 1 (node 1, ie , source)			(string)
	r2- region 2 (node 2, ie, destination)			(string)
	wt- edge weight you want between these two nodes	(float)

6. Print the graph and susceptibilities of each node:-

	g.print_graph()
	

7. For calculation of susceptibility, Decay Factor, Transport factor and weight between two edges following libraries are imported: numpy,pandas and math

8. The geography matrix of 9 cities i.e temperature,rain and humidity are initiated with certain values

9. The coordinates of regions A to I are also initiated with certain coordinates

10. The Euclidean distance is calculated between two cities
      def dist(region1, region2)
    PARAMETERS:-
    	region1
	region2

11. Minimum distance  min_dist between two cities is found by using loops

12. The susceptibility of the ith city is calculated.The susceptibility is dependent on rain, humid and temperature.
       def calc_susceptibility(i)
     PARAMETERS:
        i: stands for the ith city
	
13. The Decay Factor is calculated by taking
        Time t=1
	using np.random.random() function to find the decay constants:
	i.)tov
	ii.)c
	iii.)phi
	iv.)psi
	
14.The above parameters are passed into the function def find_decay()
	the decay factor is calculated by using the parameters mentioned in the 13th point and are put in a formula
	
15.The coefficients of Transport Factor are calculated by using np.random.normal().They are as follows:
     
     a = P(L1|L2) given the mode of transmission is air at the exact moment the disease is first detected in L2

     w = P(L1|L2) given the mode of transmission is water at the exact moment the disease is first detected in L2

     i = P(L1|L2) given insects act as a vector at the exact moment the disease is first detected in L2

     h = P(L1|L2) given humans act as a vector at the exact moment the disease is first detected in L2
  The above parameters are the converted to numpy array.
  
 16.The weight between two edges is calculated using the following function:
      def calc_weight(x, y)
      PARAMETERS:
      x:region1 coordinates
      y:region2 coordinates
      The formula is:
      V = Vn.D(Vn,t)  [Matrix multiplication is done using np.matmul]
      where Vn is called Transport Factor
      D(Vn,t) is Decay Factor
      
      

      
      
       
 
     
  

	
	
       
               
	
