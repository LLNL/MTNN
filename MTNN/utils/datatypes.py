"""Holds namedtuples definitions"""
from collections import namedtuple

rhs = namedtuple('rhs', ['W', 'b'])
level_data = namedtuple('level_data', ['R', 'P', 'W', 'b', 'rhsW', 'rhsB'])
operators = namedtuple("operators", "R_op P_op R_for_grad_op P_for_grad_op, l2reg_left_vecs, l2reg_right_vecs")
