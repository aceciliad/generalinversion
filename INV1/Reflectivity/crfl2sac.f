c
c
c  This program reformats Santmeier and Wentzel's reflectivity output
c  into a sac data file.
c
c  10/23/90   Modified to automatically enter distance in header
c                                                     -jah

      real z(1000),d(1000),a(1000),b(1000),xx(1000)
      real x(1000),rho(1000),s(131072)
      integer*4 is(131072)
      character*10 sacfil
      character*25 names
      character*1 tflag
      character*80 title
c
      write(6,90)
      read(5,10)names
      open(1,file=names)
      open(4,file='sac.output')
      rewind 1
      rewind 4
      tflag = 'n'
      write(6,93)
      read(5,*)tflag
      if(tflag.ne.'y'.and.tflag.ne.'n') tflag = 'n'
c
c nol:    no. of layers
c mdeck: top of reflectivity layer
c iss:  iss=0 (hori and vert), iss=1 (vert), iss=2 (hori)
c kom:  kom=1 (Vert) kom=2 (Hori)
c iso:  source layer  
c

      read(1,101)title
101   format(a52)
      read(1,110) nol,mdeck,iso,iss
      write(4,*) nol,mdeck,iso,iss
c
c  z(i):   depth to top of layer
c  d(i):   thickness of layer
c  a(i):   p-wave velocity of layer
c  b(i):   s-wave velocity of layer
c  rho(i): density of layer
c
      nl=nol-1
      md=mdeck+1
      if(mdeck.gt.0) then
      read(1,120) (z(i),d(i),a(i),b(i),rho(i), i=1,mdeck)
      endif
      if(md.gt.nl) go to 1
      read(1,120) (z(i),d(i),a(i),b(i),rho(i), i=md,nl)
1     read(1,140) a(nol),b(nol),rho(nol)
c
c read in the no. of receivers (nent) and receiver locations (x(i))
c 
c

      read(1,150) nent
      read(1,160) (x(i), i=1,nent)
      nseis = nent
      if(iss.eq.0) nseis = 2 * nent
      write(4,*)'no. of receivers = ',nent
      write(4,*)'distance to stations = ',(x(i),i=1,nent)
      read(1,100)vred,tmin,dt
100   format(2f10.3,f10.4)
      write(4,*)'vred= ',vred,' tmin= ',tmin,' dt = ',dt
      do 200 i = 1, nseis
      read(1,170)x(i),abal,kom,npts,balmax
      read(1,180)(is(j),j=1,npts)
      btime = abal
      if(vred.ne.0.0) btime = abal - abs(x(i))/vred
      sln = i
      do 300 j =1, npts
      s(j) = float(is(j))/10000.0
300   s(j) = balmax * ( s(j) - 5.001 ) / 4.99
      if(tflag.eq.'y') then
         do 350 j = 1, npts
350      s(j) = s(j) * ( dt * float(j-1) )**2
      endif
      if(iss.eq.0.and.kom.eq.2) then
         nx = nx + 1
         sacfil = 's000.h    '
         if(nx.lt.10) write(sacfil(4:4),'(i1)')nx
         if(nx.ge.10.and.nx.lt.100) 
     1	              write(sacfil(3:4),'(i2)')nx
         if(nx.ge.100.and.nx.lt.1000) 
     1	              write(sacfil(2:4),'(i3)')nx
	 call newhdr
	 call setfhv('b',btime,nerr)
	 call setnhv('npts',npts,nerr)
	 call setlhv('leven',.true.,nerr)
	 call setfhv('delta',dt,nerr)
	 call setfhv('dist',x(i),nerr)
         call wsac0(sacfil,xx,s,nerr)
      endif
      if(iss.eq.0.and.kom.eq.1) then
         nz = nz + 1
         sacfil = 's000.v    '
         if(nz.lt.10) write(sacfil(4:4),'(i1)')nz
         if(nz.ge.10.and.nz.lt.100) 
     1	              write(sacfil(3:4),'(i2)')nz
         if(nz.ge.100.and.nz.lt.1000) 
     1	              write(sacfil(2:4),'(i3)')nz
	 call newhdr
	 call setfhv('b',btime,nerr)
	 call setnhv('npts',npts,nerr)
	 call setlhv('leven',.true.,nerr)
	 call setfhv('delta',dt,nerr)
	 call setfhv('dist',x(i),nerr)
         call wsac0(sacfil,xx,s,nerr)
      endif
      if(iss.eq.1.or.iss.eq.2.or.iss.eq.3) then
         nz = nz + 1
	 nx = nx + 1
	 if(iss.eq.1) sacfil = 's000.v    '
	 if(iss.eq.2) sacfil = 's000.h    '
	 if(iss.eq.3) sacfil = 's000.sh   '
         if(nz.lt.10) write(sacfil(4:4),'(i1)')nz
         if(nz.ge.10.and.nz.lt.100) 
     1	              write(sacfil(3:4),'(i2)')nz
         if(nz.ge.100.and.nz.lt.1000) 
     1	              write(sacfil(2:4),'(i3)')nz
	 call newhdr
	 call setfhv('b',btime,nerr)
	 call setnhv('npts',npts,nerr)
	 call setlhv('leven',.true.,nerr)
	 call setfhv('delta',dt,nerr)
	 call setfhv('dist',x(i),nerr)
         call wsac0(sacfil,xx,s,nerr)
      endif
      write(4,*)'station location = ',x(i)
      write(4,*)'starting time of trace = ',abal
      write(4,*)'component (hor=1, ver=2): ',kom
      write(4,*)'npts = ',npts
      write(4,*)'max. amplitude = ',balmax
      read(1,190)x(i)
190   format(f15.5)
200   continue
      if(iss.eq.0.or.iss.eq.1) write(6,220)nz
      if(iss.eq.0.or.iss.eq.2) write(6,230)nx
      if(iss.eq.3)write(6,240)nx
90    format(' enter name of inputfile: ',$)
93    format(' scale each trace by time**2 (y/n): ',$)
10    format(a25)
110   format(4i5)
120   format(5f10.4)
140   format(20x,3f10.4)
150   format(i5)
160   format(7f10.3)
170   format(2f15.5,i5,/,i10,5x,e15.4)
180   format(16i5)
220   format(i5,' vert component stations output ')
230   format(i5,' hori component stations output ')
240   format(i5,' sh component stations output ')
      stop
      end
