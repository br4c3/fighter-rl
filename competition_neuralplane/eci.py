"""Batched WGS84/ECI helpers matching JSBSim's propagation frames."""
from __future__ import annotations
import torch

A_FT=20925646.32546
B_FT=20855486.5951
GM=14.0764417572e15
J2=1.08262982e-3
OMEGA=7.292115e-5

def geodetic_to_ecef(lat,lon,alt):
    a=lat.new_tensor(A_FT);b=lat.new_tensor(B_FT);e2=1-(b/a).square()
    sl,cl=torch.sin(lat),torch.cos(lat);so,co=torch.sin(lon),torch.cos(lon)
    n=a/torch.sqrt(1-e2*sl.square())
    return torch.stack(((n+alt)*cl*co,(n+alt)*cl*so,(n*(1-e2)+alt)*sl),1)

def geocentric_to_ecef(lat_gc,lon,alt):
    a=lat_gc.new_tensor(A_FT);b=lat_gc.new_tensor(B_FT)
    sl,cl=torch.sin(lat_gc),torch.cos(lat_gc);so,co=torch.sin(lon),torch.cos(lon)
    sea_level_radius=1/torch.sqrt(cl.square()/a.square()+sl.square()/b.square())
    radius=sea_level_radius+alt
    return torch.stack((radius*cl*co,radius*cl*so,radius*sl),1)

def ecef_to_geocentric(r):
    x,y,z=r.unbind(1);a=x.new_tensor(A_FT);b=x.new_tensor(B_FT)
    lon=torch.atan2(y,x);rxy=torch.sqrt(x.square()+y.square())
    radius=torch.linalg.vector_norm(r,dim=1).clamp_min(1.)
    lat=torch.atan2(z,rxy)
    sl,cl=torch.sin(lat),torch.cos(lat)
    sea_level_radius=1/torch.sqrt(cl.square()/a.square()+sl.square()/b.square())
    alt=radius-sea_level_radius
    return lat,lon,alt

def ecef_to_geodetic(r):
    x,y,z=r.unbind(1);a=x.new_tensor(A_FT);b=x.new_tensor(B_FT)
    e2=1-(b/a).square();lon=torch.atan2(y,x);p=torch.sqrt(x.square()+y.square())
    lat=torch.atan2(z,p*(1-e2))
    for _ in range(5):
        n=a/torch.sqrt(1-e2*torch.sin(lat).square())
        alt=p/torch.cos(lat).clamp_min(1e-8)-n
        lat=torch.atan2(z,p*(1-e2*n/(n+alt)))
    n=a/torch.sqrt(1-e2*torch.sin(lat).square());alt=p/torch.cos(lat).clamp_min(1e-8)-n
    return lat,lon,alt

def ned_to_ecef_matrix(lat,lon):
    sl,cl,so,co=torch.sin(lat),torch.cos(lat),torch.sin(lon),torch.cos(lon);z=torch.zeros_like(lat)
    return torch.stack((torch.stack((-sl*co,-so,-cl*co),1),
                        torch.stack((-sl*so, co,-cl*so),1),
                        torch.stack((cl,z,-sl),1)),1)

def body_to_ned_matrix(euler):
    phi,theta,psi=euler.unbind(1);sp,cp=torch.sin(phi),torch.cos(phi);st,ct=torch.sin(theta),torch.cos(theta);ss,cs=torch.sin(psi),torch.cos(psi)
    return torch.stack((torch.stack((ct*cs,sp*st*cs-cp*ss,cp*st*cs+sp*ss),1),
                        torch.stack((ct*ss,sp*st*ss+cp*cs,cp*st*ss-sp*cs),1),
                        torch.stack((-st,sp*ct,cp*ct),1)),1)

def matrix_to_euler(r):
    pitch=torch.asin((-r[:,2,0]).clamp(-1,1));roll=torch.atan2(r[:,2,1],r[:,2,2]);yaw=torch.atan2(r[:,1,0],r[:,0,0])
    return torch.stack((roll,pitch,yaw),1)

def quaternion_to_matrix(q):
    w,x,y,z=q.unbind(1);two=2.
    return torch.stack((torch.stack((1-two*(y*y+z*z),two*(x*y-z*w),two*(x*z+y*w)),1),
                        torch.stack((two*(x*y+z*w),1-two*(x*x+z*z),two*(y*z-x*w)),1),
                        torch.stack((two*(x*z-y*w),two*(y*z+x*w),1-two*(x*x+y*y)),1)),1)

def matrix_to_quaternion(r):
    # Branch-free candidates followed by selection; robust for all attitudes.
    qabs=torch.sqrt(torch.clamp(torch.stack((1+r[:,0,0]+r[:,1,1]+r[:,2,2],1+r[:,0,0]-r[:,1,1]-r[:,2,2],1-r[:,0,0]+r[:,1,1]-r[:,2,2],1-r[:,0,0]-r[:,1,1]+r[:,2,2]),1),min=0))*0.5
    w,x,y,z=qabs.unbind(1);eps=r.new_tensor(1e-9)
    cand=torch.stack((torch.stack((w,(r[:,2,1]-r[:,1,2])/(4*w).clamp_min(eps),(r[:,0,2]-r[:,2,0])/(4*w).clamp_min(eps),(r[:,1,0]-r[:,0,1])/(4*w).clamp_min(eps)),1),
                      torch.stack(((r[:,2,1]-r[:,1,2])/(4*x).clamp_min(eps),x,(r[:,0,1]+r[:,1,0])/(4*x).clamp_min(eps),(r[:,0,2]+r[:,2,0])/(4*x).clamp_min(eps)),1),
                      torch.stack(((r[:,0,2]-r[:,2,0])/(4*y).clamp_min(eps),(r[:,0,1]+r[:,1,0])/(4*y).clamp_min(eps),y,(r[:,1,2]+r[:,2,1])/(4*y).clamp_min(eps)),1),
                      torch.stack(((r[:,1,0]-r[:,0,1])/(4*z).clamp_min(eps),(r[:,0,2]+r[:,2,0])/(4*z).clamp_min(eps),(r[:,1,2]+r[:,2,1])/(4*z).clamp_min(eps),z),1)),1)
    idx=qabs.argmax(1);q=cand[torch.arange(len(r),device=r.device),idx]
    return q/torch.linalg.vector_norm(q,dim=1,keepdim=True).clamp_min(eps)

def earth_rotation(angle):
    c,s=torch.cos(angle),torch.sin(angle);z=torch.zeros_like(c);o=torch.ones_like(c)
    return torch.stack((torch.stack((c,s,z),1),torch.stack((-s,c,z),1),torch.stack((z,z,o),1)),1)

def gravity_j2(ecef):
    radius=torch.linalg.vector_norm(ecef,dim=1);sinlat=ecef[:,2]/radius
    common=1.5*ecef.new_tensor(J2)*(ecef.new_tensor(A_FT)/radius).square()
    xy=1-5*sinlat.square();zz=3-5*sinlat.square();scale=-ecef.new_tensor(GM)/radius.pow(3)
    return ecef*scale[:,None]*torch.stack((1+common*xy,1+common*xy,1+common*zz),1)
