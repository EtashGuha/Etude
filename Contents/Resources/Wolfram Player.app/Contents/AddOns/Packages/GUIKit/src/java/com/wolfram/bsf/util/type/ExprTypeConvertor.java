/*
 * @(#)ExprTypeConvertor.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.type;

import com.wolfram.jlink.Expr;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Vector;
import java.awt.Color;

/**
 * ExprTypeConvertor
 */
public class ExprTypeConvertor implements MathematicaTypeConvertor {

	public Object convert(Class from, Class to, Object obj) {

		if (to == Expr.class) {
			if (from == Expr.class) {
				return obj;
				}
	    else if (from == Boolean.class || from == boolean.class) {
	      return (Boolean.TRUE.equals(obj) ? Expr.SYM_TRUE : Expr.SYM_FALSE);
	      }
	    else if (from == Double.class || from == double.class) {
	      return new Expr(Expr.REAL, ((Double)obj).toString());
	      }
      else if (from == Float.class || from == float.class) {
        return new Expr(Expr.REAL, ((Float)obj).toString());
        }
	    else if (from == Long.class || from == long.class) {
	      return new Expr(Expr.INTEGER, ((Long)obj).toString());
	      }
	    else if (from == Integer.class || from == int.class) {
	      return new Expr(Expr.INTEGER, ((Integer)obj).toString());
	      }
	    else if (from == String.class) {
	      return new Expr( (String)obj);
	      }
	    else if (from == BigDecimal.class) {
	      return new Expr( (BigDecimal)obj);
	      }
	    else if (from == BigInteger.class) {
	      return new Expr( (BigInteger)obj);
	      }
			else if (from == Color.class) {
				float[] components = {0,0,0,0};
				((Color)obj).getRGBComponents(components);
	      return new Expr(new Expr(Expr.SYMBOL, "RGBColor"),
	        new Expr[]{new Expr(components[0]), new Expr(components[1]), new Expr(components[2])});
	      }
				
	    else {
	      return null;
	      }
    	}
		else if (from == Expr.class) {
			try {
			if (to == Expr.class) {
				return obj;
				}
			else if (to == Boolean.class || to == boolean.class) {
				return ((Expr)obj).equals(Expr.SYM_TRUE) ? Boolean.TRUE : Boolean.FALSE;
				}
			else if (to == Double.class || to == double.class) {
				return new Double(((Expr)obj).asDouble());
				}
      else if (to == Float.class || to == float.class) {
        return new Float(((Expr)obj).asDouble());
        }
			else if (to == Long.class || to == long.class) {
				return new Long(((Expr)obj).asLong());
				}
			else if (to == Integer.class || to == int.class) {
				return new Integer(((Expr)obj).asInt());
				}
      else if (to == Byte.class || to == byte.class) {
        return new Byte((byte)(((Expr)obj).asInt()));
        }
			else if (to == String.class) {
				if (((Expr)obj).stringQ()) return ((Expr)obj).asString();
				else return ((Expr)obj).toString();
				}
      else if (to == Integer[].class || to == int[].class) {
        Expr e = (Expr)obj;
        int count = e.length();
        int[] objs = new int[count];
        for (int i = 0; i < count; ++i) {
          objs[i] = ((Integer)convert(Expr.class, Integer.class, e.part(i+1))).intValue();
          }
        return objs;
        }
      else if (to == Double[].class || to == double[].class) {
        Expr e = (Expr)obj;
        int count = e.length();
        double[] objs = new double[count];
        for (int i = 0; i < count; ++i) {
          objs[i] = ((Double)convert(Expr.class, Double.class, e.part(i+1))).doubleValue();
          }
        return objs;
        }
     else if (to == Byte[].class || to == byte[].class) {
        Expr e = (Expr)obj;
        int count = e.length();
        byte[] objs = new byte[count];
        for (int i = 0; i < count; ++i) {
          objs[i] = ((Byte)convert(Expr.class, Byte.class, e.part(i+1))).byteValue();
          }
        return objs;
        }
      else if (to == String[].class) {
        Expr e = (Expr)obj;
        int count = e.length();
        String[] objs = new String[count];
        for (int i = 0; i < count; ++i) {
          objs[i] = (String)convert(Expr.class, String.class, e.part(i+1));
          }
        return objs;
        }
			else if (to == BigDecimal.class) {
				return ((Expr)obj).asBigDecimal();
				}
			else if (to == BigInteger.class) {
				return ((Expr)obj).asBigInteger();
				}
				
      else if (Color.class.isAssignableFrom(to)) {
        Color c = null;
        Expr e = (Expr)obj;
        if (e.head().equals(new Expr(Expr.SYMBOL, "RGBColor"))) {
          c = new Color((float)e.part(1).asDouble(), (float)e.part(2).asDouble(), (float)e.part(3).asDouble());
          }
        else if (e.head().equals(new Expr(Expr.SYMBOL, "GrayLevel"))) {
          float grey = (float)e.part(1).asDouble();
          c = new Color(grey,grey,grey);
          }
        else if (e.head().equals(new Expr(Expr.SYMBOL, "Hue"))) {
          float h = (float)e.part(1).asDouble();
          float s = (float)1.0;
          float b = (float)1.0;
          if (e.length() > 1) {
            s = (float)e.part(2).asDouble();
            b = (float)e.part(3).asDouble();
            }
          c = Color.getHSBColor(h, s, b);
          }
        // TODO add some sort of support for CMYKColor[]
        return c;
        }
					
      else if (to == Object[][].class) {
        Expr e = (Expr)obj;
        int count = e.length();
        Object[][] objs = new Object[count][];
        for (int i = 0; i < count; ++i) {
          objs[i] = (Object[])convert(Expr.class, Object[].class, e.part(i+1));
          }
        return objs;
        }
		  else if (to == Object[].class) {
		  	Expr e = (Expr)obj;
		  	int count = e.length();
		  	Object[] objs = new Object[count];
		  	for (int i = 0; i < count; ++i) {
		  		objs[i] = convert(Expr.class, Object.class, e.part(i+1));
		  		}
		  	return objs;
		    }
      else if (to == Vector.class) {
        Expr e = (Expr)obj;
        int count = e.length();
        Vector vec = new Vector(count);
        for (int i = 0; i < count; ++i) {
          vec.addElement(convert(Expr.class, Object.class, e.part(i+1)) );
          }
        return vec;
        }
		  else if (to == Object.class) {
		  	Expr e = (Expr)obj;
        if (e.equals(Expr.SYM_TRUE) || e.equals(Expr.SYM_FALSE)) return (e.equals(Expr.SYM_TRUE) ? Boolean.TRUE : Boolean.FALSE);
				else if (e.stringQ() || e.symbolQ()) return e.asString();
				else if (e.matrixQ(Expr.INTEGER)) return e.asArray(Expr.INTEGER, 2);
				else if (e.matrixQ(Expr.REAL)) return e.asArray(Expr.REAL, 2);
				else if (e.vectorQ(Expr.INTEGER)) return e.asArray(Expr.INTEGER, 1);
				else if (e.vectorQ(Expr.REAL)) return e.asArray(Expr.REAL, 1);
				else if (e.listQ()) return convert(Expr.class, Object[].class, e);
				else if (e.bigDecimalQ()) return e.asBigDecimal();
				else if (e.bigIntegerQ()) return e.asBigInteger();
				else if (e.integerQ()) return new Integer(e.asInt());
				else if (e.realQ()) return new Double(e.asDouble());
		  	return obj;
		    }
			else {
				return null;
				}
			}
		 catch(Exception e) {return null;}
			}
		else
			return null;
	}

  public Object convertExprAsContent(Expr e) {
  	if (e.bigDecimalQ()) {
  		return convert(Expr.class, BigDecimal.class, e);
  		}
		else if (e.bigIntegerQ()) {
			return convert(Expr.class, BigInteger.class, e);
			}
		else if (e.stringQ()) {
			return convert(Expr.class, String.class, e);
			}
		else if (e.equals(Expr.SYM_TRUE) || e.equals(Expr.SYM_FALSE)) {
			return convert(Expr.class, Boolean.class, e);
			}
		else if (e.realQ()) {
			return convert(Expr.class, Double.class, e);
			}
		else if (e.integerQ()) {
			return convert(Expr.class, Integer.class, e);
			}
  	return e;
  	}
  	
  // Not implemented or used anywhere
  public String getCodeGenString () {
    return "";
    }
  
}

