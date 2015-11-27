#include<iostream>
#include<map>
#include<vector>
#include<queue>
#include<stdlib.h>
#include<algorithm>
#include<ctime>

#define NODES 5000 
#define G2_ADJ_SIZE 1000 

using namespace std;

bool myfind(vector<int> a, int n)
{
	for(int i=0;i<a.size();i++)
	{
		if(a[i]==n)
			return true;
	}
	return false;
}

int max(int a,int b)
{
  if(a>=b)
    return a;
  else
    return b;
}

/******************************************************************************/
//Max Bandwidth - Dijkstra
void max_bandwidth(map<int,vector<int> > G, int s, int t, map<int,vector<int> > G_weight)
{
	for(int i=1;i<=NODES;i++)
	{
		for(int j=0;j<G[i].size();j++)
		{
			if(G[i][j]==i)
			{
				G[i].erase(G[i].begin() + j);
				G_weight[i].erase(G_weight[i].begin() + j);
			}
		}		
	}

	clock_t begin = clock();
	int status[NODES+1];
	int cap[NODES+1];
	int dad[NODES+1];
	int fringe_exists=0;
	for(int i=1;i<=NODES;i++)
	{
		status[i] = -1;//-1=>unseen, 0=>fringe, 1=>in-tree
		cap[i]=0;
		dad[i]=-1;
	}
	status[s]=1;//in-tree source node
	for(int i=0;i<G[s].size();i++)//For each edge [s,w]
	{
		fringe_exists=1;
		status[G[s][i]]=0;//status[w]=fringe
		cap[G[s][i]]=G_weight[s][i];
		dad[G[s][i]] = s;
	}
	
	while(fringe_exists)
	{
		//find largest fringe
		int i;
		for(i=1;i<=NODES;i++)
		{
			if(status[i]==0)
			break;
		}
		if(i==NODES+1)
		{
			break;
		}
		int v=i;
		for(i=1;i<=NODES;i++)
		{
			if(status[i]==0 && cap[i]>cap[v])
			v=i;
		}
		status[v]=1;//status[v]=in-tree
		for(int i=0;i<G[v].size();i++)
		{
			int w = G[v][i];
			if(status[w]==-1)//if w=unseen
			{
				status[w]=0;//status[w]=fringe
				dad[w]=v;
				cap[w]=min(cap[v],G_weight[v][i]);
			}
			else if(status[w]==0 && cap[w]<min(cap[v],G_weight[v][i]))
			{
				dad[w]=v;
				cap[w]=min(cap[v],G_weight[v][i]);
			}
		}
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / (double)CLOCKS_PER_SEC;

	dad[s]=-1;//For cycles
	cout<<"The max bw path is:"<<endl;
	while(t!=s)
	{
		if(t==-1)
		break;
		cout<<t<<" ";
		t=dad[t];
	}
	cout<<s<<endl;
	cout<<"Elapsed secs:"<<elapsed_secs<<endl;
}

/*******************************************************************/
int N[NODES+1];
void max_heapify(vector<int> &A,int i,vector<int> &H)
{
  int l=2*i;
  int r=2*i + 1;
	int largest;
  if(l<=A.size()-1 && A[l]>A[i])
	largest = l;
	else
	largest = i;
	if(r<=A.size()-1 && A[r]>A[largest])
	largest = r;
	if(largest != i)
	{
      int temp = N[H[largest]];
      N[H[largest]] = N[H[i]];
      N[H[i]] = temp;

		  temp = H[largest];
      H[largest]=H[i];
      H[i]=temp;


      temp = A[i];
      A[i]=A[largest];
      A[largest]=temp;
			max_heapify(A,largest,H);
	}
}

/*********************************************/
int heap_max(int A[NODES+1], vector<int> H)
{
	if(H.size()<=1)//0,<list> -> 0th position in H contains 0 always
	return 0;
  return H[1];
}

/*****************************************************************/
vector<int> build_max_heap(int a[NODES+1])
{
  vector<int> H;
  vector<int> A;
  vector<int>::iterator it;
	a[0]=0;
	A.push_back(0);
  
  for(int i=1;i<=NODES;i++)
    A.push_back(a[i]);	
  
  H.push_back(0);
  N[0] = 0;
  
  for(int i=1;i<=NODES;i++)
  {
    H.push_back(i);
    N[i] = i;
  }
  for(int i=(A.size()-1)/2;i>=1;i--)
  {
    max_heapify(A,i,H);
  }
  for(int i=1;i<H.size();i++)
  {
		if(a[H[i]]==0)
		{
      it = H.begin() + i;
			H.erase(it);
      N[*it] = 0;
			i--;
		}
	}
  return H;
}


/******************************************************************/
void heap_insert(int A[NODES+1],int n, vector<int> &H)
{
	H.push_back(n);
  N[n] = H.size() - 1;
	int h=H.size()-1;
	//dont call max_heapify from here, it will change A
  while(h>1 && A[H[h]]>A[H[h/2]])
  {
    int temp = N[H[h]];
    N[H[h]] = N[H[h/2]];
    N[H[h/2]] = temp;

    temp = H[h];
    H[h]=H[h/2];
    H[h/2] = temp;

    h=h/2;
  }
}

/****************************************************************/
void heap_delete(int A[NODES+1], int index_in_H, vector<int> &H)
{
	int h=index_in_H;
  vector<int>::iterator it = H.begin()+(H.size()-1);

  N[H[h]] = N[H[H.size()-1]];
	H[h] = H[H.size()-1];
	H.erase(H.begin()+(H.size()-1));
  N[*it] = 0;
	
  if(H.size()==2)
  	return;
	
  if(h>1 && A[H[h]]>A[H[h/2]])//Push up
	{
		while(h>1 && A[H[h]]>A[H[h/2]])
		{
		  int temp = N[H[h]];
		  N[H[h]]=N[H[h/2]];
		  N[H[h/2]] = temp;
		  
      temp = H[h];
		  H[h]=H[h/2];
		  H[h/2] = temp;

		  h=h/2;
		}
	}
	else 
	{
		int l=2*index_in_H;
		int r=2*index_in_H + 1;
		int largest;
		if(l<=H.size()-1 && A[H[l]]>A[H[index_in_H]])
		largest = l;
		else
		largest = index_in_H;
		if(r<=H.size()-1 && A[H[r]]>A[H[largest]])
		largest = r;
		while(largest != index_in_H)
		{
			int temp = N[H[largest]];
	    N[H[largest]] = N[H[index_in_H]];
	    N[H[index_in_H]] = temp;
			
      temp = H[largest];
	    H[largest]=H[index_in_H];
	    H[index_in_H]=temp;

			index_in_H = largest;
			l = 2*largest;
			r = 2*largest + 1;
			if(l>H.size()-1 && r>H.size()-1)
  			break;
			if(l<=H.size()-1 && A[H[l]]>A[H[index_in_H]])
	  		largest = l;
			else
		  	largest = index_in_H;
			if(r<=H.size()-1 && A[H[r]]>A[H[largest]])
			  largest = r;
		}
	}
}

/*************************************************************/
//Max Bandwidth - Dijkstra + Heap
void max_bw_heap(map<int,vector<int> > G, int s, int t, map<int,vector<int> > G_weight)
{
	for(int i=1;i<=NODES;i++)
	{
		for(int j=0;j<G[i].size();j++)
		{
			if(G[i][j]==i)
			{
				G[i].erase(G[i].begin() + j);
				G_weight[i].erase(G_weight[i].begin() + j);
			}
		}		
	}
	
  clock_t begin = clock();
	int status[NODES+1];
	int cap[NODES+1];
	int dad[NODES+1];
	int fringe_exists=0;
	vector<int> mheap;

	for(int i=1;i<=NODES;i++)
	{
		status[i] = -1;//-1=>unseen, 0=>fringe, 1=>in-tree
		cap[i]=0;
		dad[i]=-1;
	}
	status[s]=1;//in-tree source node
	for(int i=0;i<G[s].size();i++)//For each edge [s,w]
	{
		fringe_exists=1;
		status[G[s][i]]=0;//status[w]=fringe
		cap[G[s][i]]=G_weight[s][i];
		dad[G[s][i]] = s;
	}
	mheap = build_max_heap(cap);	

	while(fringe_exists)
	{
		//find largest fringe
		int i=heap_max(cap, mheap);
		if(i==0)//No fringes left
		{
			break;
		}
		heap_delete(cap,1,mheap);

		int v=i;
		status[v]=1;//status[v]=in-tree
		for(int i=0;i<G[v].size();i++)
		{
			int w = G[v][i];
			if(status[w]==-1)//if w=unseen
			{
				status[w]=0;//status[w]=fringe
				dad[w]=v;
				cap[w]=min(cap[v],G_weight[v][i]);
				heap_insert(cap,w,mheap);
			}
			else if(status[w]==0 && cap[w]<min(cap[v],G_weight[v][i]))
			{
				dad[w]=v;
				cap[w]=min(cap[v],G_weight[v][i]);
        int pos = N[w];

				if(pos!=0)
				{
				heap_delete(cap,pos,mheap);
				heap_insert(cap,w,mheap);
				}
			}
		}

	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / (double)CLOCKS_PER_SEC;
	
  dad[s]=-1;//For cycles
	cout<<"The max bw path is:"<<endl;
	while(t!=s)
	{
		if(t==-1)
		break;
		cout<<t<<" ";
		t=dad[t];
	}
	cout<<s<<endl;
	cout<<"Elapsed secs:"<<elapsed_secs<<endl;
}

/*********************/
class edge
{
	public:
	int source;
	int dest;
	int wt;
};

/**********************/
class CompareMake
{
public:
    // Compare two edges
    bool operator()(edge* x, edge* y)
    {
        return x->wt < y->wt;
    }
};

/****************************************/
int find_krus(int dad[NODES+1],int x)
{
	int w = x;
	vector<int> S;
	while(dad[w]!=0)
	{
		S.push_back(w);
		w=dad[w];
	}
	for(int i=0;i<S.size();i++)
	{
		dad[S[i]]=w;
	}
	return w;
}

/*******************************************************************/
int union_krus(int r1,int r2, int rank[NODES+1],int dad[NODES+1])
{
	if(rank[r1]>rank[r2])
	dad[r2]=r1;
	else if(rank[r2]>rank[r1])
	dad[r1]=r2;
	else if(rank[r1]==rank[r2])
	{
		dad[r2]=r1;
		rank[r1]++;
	}
}
static int flag = 0;
/**********************************************/
void node_dfs(int s, int t, int vertex_colors[NODES+1], map<int, vector<int> > &adj)
{
  vertex_colors[s] = 1;
  if(s==t)
  {
    flag = 1;
    cout<<s<<" ";
    vertex_colors[s]=2;
    return;
  }
  vector<int> child_vertices = adj[s];
  int child_count = adj[s].size();
  
  for(int i=0; i<child_count; i++)
  {
    if(vertex_colors[child_vertices[i]] == 0)
    {
      node_dfs(child_vertices[i], t, vertex_colors, adj);
    }
    if( flag == 1)
    {
      cout<<s<<" ";
      vertex_colors[s] = 2;
      return;
    }
  }

  vertex_colors[s] = 2;
}

/***********************************************/
void dfs(int s, int t, map<int, vector<int> > adj)
{
  int *vertex_colors = new int[adj.size()];
  map<int, vector<int> >::iterator it = adj.begin();

  for(; it!=adj.end(); it++)
  {
    vertex_colors[it->first] = 0;
  }

  flag = 0;
  node_dfs(s, t, vertex_colors, adj);
  cout<<endl;
}

/******************************************************************/
//Max Bandwidth - Kruskal's
void max_bw_kruskal(map<int,vector<int> > G, int s, int t, map<int,vector<int> > G_weight)
{
	for(int i=1;i<=NODES;i++)
	{
		for(int j=0;j<G[i].size();j++)
		{
			if(G[i][j]==i)
			{
				G[i].erase(G[i].begin() + j);
				G_weight[i].erase(G_weight[i].begin() + j);
			}
		}	
	}	
  
  clock_t begin = clock();
	vector<edge*> edges;
	for(int i=1;i<=NODES;i++)
	{
		for(int j=0;j<G[i].size();j++)
		{
			edge *e = new edge();
			e->source = i;
			e->dest = G[i][j];
			e->wt = G_weight[i][j];
			edges.push_back(e);
		}		
	}
	make_heap(edges.begin(),edges.end(),CompareMake());
	sort_heap(edges.begin(),edges.end(),CompareMake());
	vector<edge*> sorted_edges;
  for (vector<edge*>::reverse_iterator it=edges.rbegin(); it!=edges.rend(); ++it)
    sorted_edges.push_back(*it);
	
	//Makeset
	int dad[NODES+1];
	int rank[NODES+1];
	for(int i=1;i<=NODES;i++)
	{
		dad[i]=0;
		rank[i]=0;
	}

	vector<edge*> soln_edges;
  map<int, vector<int> > adj;
	for(int i=0;i<sorted_edges.size();i++)
	{
		int v = sorted_edges[i]->source;
		int w = sorted_edges[i]->dest;
		int r1 = find_krus(dad,v);//Find(v)
		int r2 = find_krus(dad,w);//Find(w)
		if(r1!=r2)
		{
			soln_edges.push_back(sorted_edges[i]);
			union_krus(r1,r2,rank,dad);
		}
	}

  //Getting Max BW Path by DFS
  for(vector<edge*>::iterator it=soln_edges.begin(); it!=soln_edges.end(); it++)
  {
    adj[(*it)->source].push_back((*it)->dest);
    adj[(*it)->dest].push_back((*it)->source);
  }

  dfs(s, t, adj);
  clock_t end = clock();
  double estimated = double(end-begin)/(double)CLOCKS_PER_SEC;
  cout<<"Elapsed secs:"<<estimated<<endl;
}

/*******************************************************************/
int main()
{
  map<int,vector<int> > G1;//has edge cycles to same node and may have multiple edges between 2 nodes	
  map<int,vector<int> > G2;//has no cycles and no multiple edges between two nodes
	map<int,vector<int> > G1_weight;
	map<int,vector<int> > G2_weight;
	vector<int> unique;//counts unique adjacent nodes from a node in G2

  srand(time(NULL)); // randomize seed


//For G1
	cout<<"Generating G1... "<<endl;
  clock_t begin = clock();
  for(int i=1;i<=NODES;i++)
  {
    if(G1.find(i) == G1.end())
    {
      vector<int> v;
      G1[i]=v;
    }
    while((G1[i].size())<6)
    {
			int in = rand()%NODES + 1;
			while(in==i && G1[i].size()==5)//Cycles check//Necessary when space left for one edge
				in = rand()%NODES + 1;
		  if(G1.find(in) == G1.end())
		  {
		    vector<int> v;
		    G1[in]=v;
		  }
			if(G1[in].size()<6)
			{
				{
					G1[in].push_back(i);
					G1[i].push_back(in);
				}
			}
    }
  }
  clock_t end = clock();
	double elapsed_secs = double(end - begin) / (double)CLOCKS_PER_SEC;
	cout<<"Elapsed secs:"<<elapsed_secs<<endl;
  for(int i=1;i<=NODES;i++)
  { 
		vector<int> v;
		G1_weight[i] = v;
    for(int j=0;j<G1[i].size();j++)
    {
			G1_weight[i].push_back(rand()%100 + 1);//Pushing edge weights to G1
		}
  }


//For G2
//initialize unique node count to zero
  for(int i=1;i<=NODES;i++)
	unique.push_back(0);
	unique.push_back(0);

	cout<<"Generating G2... "<<endl;
  begin = clock();
  for(int i=1;i<=NODES;i++)
  {
    if(G2.find(i) == G2.end())
    {
      vector<int> v;
      G2[i]=v;
    }
    while(unique[i]<G2_ADJ_SIZE)
    {
			int in = rand()%NODES + 1;
		  if(G2.find(in) == G2.end())
		  {
		    vector<int> v;
		    G2[in]=v;
		  }
			//if(unique[in]<G2_ADJ_SIZE)
			{
				if(in!=i && myfind(G2[i],in)==false)//one edge between x and y vertices
				{
					unique[i]++;
					if(myfind(G2[in],i)==false) 
					unique[in]++;
				}
				G2[in].push_back(i);
				G2[i].push_back(in);
			}
    }
  }
  end = clock();
	elapsed_secs = double(end - begin) / (double)CLOCKS_PER_SEC;
	cout<<"Elapsed secs:"<<elapsed_secs<<endl;

  //Print G2
  for(int i=1;i<=NODES;i++)
  {
		vector<int> v;
		G2_weight[i] = v; 
    for(int j=0;j<G2[i].size();j++)
    {
			G2_weight[i].push_back(rand()%100 + 1);
		}
  }

	int source, destination;
	source = 	rand()%NODES+1;
	destination = rand()%NODES+1;
	cout<<"s = "<<destination<<", t = "<<source<<endl;

  cout<<endl<<"Dijkstra G1"<<endl;
  cout<<"-----------------"<<endl;
	max_bandwidth(G1, source, destination, G1_weight);

	cout<<endl<<"Dijkstra heap G1"<<endl;
  cout<<"-----------------"<<endl;
	max_bw_heap(G1, source, destination, G1_weight);

	cout<<endl<<"Kruskal G1"<<endl;
  cout<<"-----------------"<<endl;
	cout<<"The max bw path is:"<<endl;
	max_bw_kruskal(G1, source, destination, G1_weight);

	
  cout<<endl<<"Dijkstra G2"<<endl;
  cout<<"-----------------"<<endl;
	max_bandwidth(G2, source, destination, G2_weight);

	cout<<endl<<"Dijkstra Heap G2"<<endl;
  cout<<"-----------------"<<endl;
	max_bw_heap(G2, source, destination, G2_weight);

	cout<<endl<<"Kruskal G2"<<endl;
  cout<<"-----------------"<<endl;
	cout<<"The max bw path is:"<<endl;
	max_bw_kruskal(G2, source, destination, G2_weight);

	return 0;
}
