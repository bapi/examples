import importlib.util
import sys
import torch
import math


def tensor_part_add(t1, t2, start_pos, end_pos, factor):
  if t1.size() != t2.size():
    print("Tensors need to be of the same shapes!")
    return
  n1 = t1.numel()
  positions = end_pos - start_pos + 1
  if n1 < positions or n1 < end_pos+1 :
    print("Tensors' size does not match with positions.", "start = ", start_pos, "stop = ", end_pos, "positions = ", positions)
    return

  dim = len(t1.size())
  if dim == 1:
    tensor_part_add1(t1, t2, start_pos, end_pos, factor)
  if dim == 2:
    tensor_part_add2(t1, t2, start_pos, end_pos, positions, factor)
  elif dim == 3:    
    tensor_part_add3(t1, t2, start_pos, end_pos, positions, factor)
  elif dim == 4:    
    tensor_part_add4(t1, t2, start_pos, end_pos, positions, factor)
  elif dim == 5:    
    tensor_part_add5(t1, t2, start_pos, end_pos, positions, factor)
  elif dim == 6:    
    tensor_part_add6(t1, t2, start_pos, end_pos, positions, factor)
  elif dim == 7:    
    tensor_part_add7(t1, t2, start_pos, end_pos, positions, factor)
  elif dim == 8:    
    tensor_part_add8(t1, t2, start_pos, end_pos, positions, factor)
  elif dim == 9:    
    tensor_part_add9(t1, t2, start_pos, end_pos, positions, factor)

def tensor_part_add1(t1, t2, start_pos, end_pos, factor):
  t1[start_pos:end_pos + 1].add_(factor, t2[start_pos:end_pos + 1])
  
def tensor_part_add2(t1, t2, start_pos, end_pos, positions, factor):
  s = t1.size(1)
  if t1.size(0) == 1 or start_pos + positions <= s:
    t1[0][start_pos:end_pos + 1].add_(factor, t2[0][start_pos:end_pos + 1])
  else:    
    i1 = int(start_pos / s)
    j1 = int(start_pos % s)
    i2 = int(end_pos / s)
    j2 = int(end_pos % s)
    t1[i1][j1:s].add_(factor, t2[i1][j1:s])
    t1[i2][0:j2 + 1].add_(factor, t2[i2][0:j2 + 1])
    if i2 - i1 > 1:
      t1[i1+1:i2].add_(factor, t2[i1+1:i2])

  
def tensor_part_add3(t1, t2, start_pos, end_pos, positions, factor):
  s = t1.size(1)*t1.size(2)
  if t1.size(0) == 1 or start_pos + positions <= s:
    tensor_part_add2(t1[0], t2[0], start_pos, end_pos, positions, factor)
  else:    
    i1 = int(start_pos / s)
    j1 = int(start_pos % s)
    i2 = int(end_pos / s)
    j2 = int(end_pos % s)
    tensor_part_add2(t1[i1], t2[i1], j1, s-1, s-j1, factor)
    tensor_part_add2(t1[i2], t2[i2], 0, j2, j2+1, factor)
    if i2 - i1 > 1:
      t1[i1+1:i2].add_(factor, t2[i1+1:i2])

def tensor_part_add4(t1, t2, start_pos, end_pos, positions, factor):
  s = t1.size(1)*t1.size(2)*t1.size(3)
  if t1.size(0) == 1 or start_pos + positions <= s:
    tensor_part_add3(t1[0], t2[0], start_pos, end_pos, positions, factor)
  else:    
    i1 = int(start_pos / s)
    j1 = int(start_pos % s)
    i2 = int(end_pos / s)
    j2 = int(end_pos % s)
    tensor_part_add3(t1[i1], t2[i1], j1, s-1, s-j1, factor)
    tensor_part_add3(t1[i2], t2[i2], 0, j2, j2+1, factor)
    if i2 - i1 > 1:
      t1[i1+1:i2].add_(factor, t2[i1+1:i2])

def tensor_part_add5(t1, t2, start_pos, end_pos, positions, factor):
  s = t1.size(1)*t1.size(2)*t1.size(3)*t1.size(4)
  if t1.size(0) == 1 or start_pos + positions <= s:
    tensor_part_add4(t1[0], t2[0], start_pos, end_pos, positions, factor)
  else:    
    i1 = int(start_pos / s)
    j1 = int(start_pos % s)
    i2 = int(end_pos / s)
    j2 = int(end_pos % s)
    tensor_part_add4(t1[i1], t2[i1], j1, s-1, s-j1, factor)
    tensor_part_add4(t1[i2], t2[i2], 0, j2, j2+1, factor)
    if i2 - i1 > 1:
      t1[i1+1:i2].add_(factor, t2[i1+1:i2])


def tensor_part_add6(t1, t2, start_pos, end_pos, positions, factor):
  s = t1.size(1)*t1.size(2)*t1.size(3)*t1.size(4)*t1.size(5)
  if t1.size(0) == 1 or start_pos + positions <= s:
    tensor_part_add5(t1[0], t2[0], start_pos, end_pos, positions, factor)
  else:    
    i1 = int(start_pos / s)
    j1 = int(start_pos % s)
    i2 = int(end_pos / s)
    j2 = int(end_pos % s)
    tensor_part_add5(t1[i1], t2[i1], j1, s-1, s-j1, factor)
    tensor_part_add5(t1[i2], t2[i2], 0, j2, j2+1, factor)
    if i2 - i1 > 1:
      t1[i1+1:i2].add_(factor, t2[i1+1:i2])



def tensor_part_add7(t1, t2, start_pos, end_pos, positions, factor):
  s = t1.size(1)*t1.size(2)*t1.size(3)*t1.size(4)*t1.size(5)*t1.size(6)
  if t1.size(0) == 1 or start_pos + positions <= s:
    tensor_part_add6(t1[0], t2[0], start_pos, end_pos, positions, factor)
  else:    
    i1 = int(start_pos / s)
    j1 = int(start_pos % s)
    i2 = int(end_pos / s)
    j2 = int(end_pos % s)
    tensor_part_add6(t1[i1], t2[i1], j1, s-1, s-j1, factor)
    tensor_part_add6(t1[i2], t2[i2], 0, j2, j2+1, factor)
    if i2 - i1 > 1:
      t1[i1+1:i2].add_(factor, t2[i1+1:i2])


def tensor_part_add8(t1, t2, start_pos, end_pos, positions, factor):
  s = t1.size(1)*t1.size(2)*t1.size(3)*t1.size(4)*t1.size(5)*t1.size(6)*t1.size(7)
  if t1.size(0) == 1 or start_pos + positions <= s:
    tensor_part_add7(t1[0], t2[0], start_pos, end_pos, positions, factor)
  else:    
    i1 = int(start_pos / s)
    j1 = int(start_pos % s)
    i2 = int(end_pos / s)
    j2 = int(end_pos % s)
    tensor_part_add7(t1[i1], t2[i1], j1, s-1, s-j1, factor)
    tensor_part_add7(t1[i2], t2[i2], 0, j2, j2+1, factor)
    if i2 - i1 > 1:
      t1[i1+1:i2].add_(factor, t2[i1+1:i2])

def tensor_part_add9(t1, t2, start_pos, end_pos, positions, factor):
  s = t1.size(1)*t1.size(2)*t1.size(3)*t1.size(4)*t1.size(5)*t1.size(6)*t1.size(7)*t1.size(8)
  if t1.size(0) == 1 or start_pos + positions <= s:
    tensor_part_add8(t1[0], t2[0], start_pos, end_pos, positions, factor)
  else:    
    i1 = int(start_pos / s)
    j1 = int(start_pos % s)
    i2 = int(end_pos / s)
    j2 = int(end_pos % s)
    tensor_part_add8(t1[i1], t2[i1], j1, s-1, s-j1, factor)
    tensor_part_add8(t1[i2], t2[i2], 0, j2, j2+1, factor)
    if i2 - i1 > 1:
      t1[i1+1:i2].add_(factor, t2[i1+1:i2])
