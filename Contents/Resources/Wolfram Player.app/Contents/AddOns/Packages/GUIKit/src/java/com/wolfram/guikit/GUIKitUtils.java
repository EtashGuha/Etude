/*
 * @(#)GUIKitUtils.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Image;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.image.ImageProducer;
import java.beans.*;
import java.io.*;
import java.lang.reflect.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.net.URL;
import java.util.*;

import javax.swing.KeyStroke;

import org.w3c.dom.*;

import com.wolfram.guikit.event.GUIKitEventProcessor;
import com.wolfram.guikit.event.GUIKitEventScriptProcessor;
import com.wolfram.guikit.layout.GUIKitLayoutInfo;
import com.wolfram.guikit.type.GUIKitTypedObject;
import com.wolfram.guikit.type.GUIKitTypedObjectInfoSet;
import com.wolfram.jlink.Expr;
import com.wolfram.jlink.LoopbackLink;
import com.wolfram.jlink.MathLink;
import com.wolfram.jlink.MathLinkException;
import com.wolfram.jlink.MathLinkFactory;

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.bsf.util.*;
import com.wolfram.bsf.util.concurrent.InvokeMode;
import com.wolfram.bsf.util.event.MathematicaEventAdapterProxy;
import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;
import com.wolfram.bsf.util.type.TypedObjectFactory;

/**
 * GUIKitUtils
 */
public class GUIKitUtils {

  public static String GUI_XMLFORMAT = "GUIKitXML";
  
  public static String PI_GUI_XMLFORMAT = GUI_XMLFORMAT;
  
  public static String ELEM_ARGS = "args";

  public static String ELEM_WIDGET = "widget";
	public static String ELEM_GROUP = "group";
	public static String ELEM_SPACE = "space";
	public static String ELEM_FILL = "fill";
	public static String ELEM_ALIGN = "align";
	
	public static String ELEM_LAYOUT = "layout"; 
	public static String ELEM_GROUPING = "grouping"; 
  public static String ELEM_SPLIT = "split"; 
  public static String ELEM_TABS = "tabs"; 
	public static String ELEM_ALIGNMENT = "alignment"; 
	public static String ELEM_STRETCHING = "stretching"; 
	public static String ELEM_SPACING = "spacing"; 
	public static String ELEM_BORDER = "border"; 
	
  public static String ELEM_BINDEVENT = "bindevent";
  public static String ELEM_PROPERTY = "property";
  public static String ELEM_INVOKEMETHOD = "invokemethod";
  public static String ELEM_SCRIPT = "script";
  
  public static String ELEM_STRING = "string";
  public static String ELEM_INTEGER = "integer";
  public static String ELEM_DOUBLE = "double";
      
  public static String ELEM_NULL = "null";
  public static String ELEM_TRUE = "true";
  public static String ELEM_FALSE = "false";
  
  public static String ELEM_EXPOSE = "expose";

	public static String ATT_LAYOUT = "layout";
  public static String ATT_ORIENT = "orient";
  public static String ATT_ROOT = "root";
	public static String ATT_TYPE = "type";
	public static String ATT_FROM = "from";
	public static String ATT_TO = "to";
	public static String ATT_TITLE = "title";
	public static String ATT_COLOR = "color";
	public static String ATT_TOP = "top";
	public static String ATT_BOTTOM = "bottom";
	public static String ATT_LEFT = "left";
	public static String ATT_RIGHT = "right";
	
  public static String ATT_CLASS = "class";
  public static String ATT_FILTER = "filter";
  public static String ATT_ID = "id";
  public static String ATT_AS = "as";
  public static String ATT_INDEX = "index";
  public static String ATT_NAME = "name";
  public static String ATT_VALUE = "value";
  public static String ATT_SRC = "src";
  public static String ATT_REF = "ref";
  public static String ATT_TARGET = "target";
  public static String ATT_LANGUAGE = "language";

  public static String ATT_INVOKETHREAD = "invokeThread";
  public static String ATT_INVOKETHREAD_DEFAULT = "Current";
  public static String ATT_INVOKEWAIT = "invokeWait";
  public static String ATT_INVOKEWAIT_DEFAULT = "Automatic";
  
  public static String ATTVAL_THIS = "this";

  public static String ATTVAL_LANGUAGE_GUI_XMLFORMAT = GUI_XMLFORMAT;
  public static String ATTVAL_LANGUAGE_XML = "xml";
  
	private static Map objectInfoSetCache = new WeakHashMap(100);
	
  public static Image createImage(Object data) {
    Image image = null;
    try {
      if (data instanceof String)
        image = Toolkit.getDefaultToolkit().createImage(
          ((String)data).getBytes("ISO-8859-1"));
      else if (data instanceof URL)
        image = Toolkit.getDefaultToolkit().createImage((URL)data);
      else if (data instanceof ImageProducer)
        image = Toolkit.getDefaultToolkit().createImage((ImageProducer)data);
      }
    catch (Exception e) {}
    return image;
    }
  
  // For now using the reflection only version til any issues arise
  // More properties are available not sure if it makes a diff for events
  // but since we will cache them we should share them
  
	private static GUIKitTypedObjectInfoSet getObjectInfoSet(GUIKitTypedObject obj) throws GUIKitException {
		if (obj == null) return null;
    GUIKitTypedObjectInfoSet infoSet = (GUIKitTypedObjectInfoSet)objectInfoSetCache.get(obj.type);
		if (infoSet == null) {
			try {
			  infoSet = new GUIKitTypedObjectInfoSet(
          Introspector.getBeanInfo(obj.type, Introspector.IGNORE_ALL_BEANINFO));
				}
			catch(IntrospectionException introspectionexception) {
				throw new GUIKitException(GUIKitException.REASON_GET_OBJECTINFO_ERROR,
					"Exception while getting object info: " + introspectionexception);
			  }
      if (infoSet != null)
			  objectInfoSetCache.put(obj.type, infoSet);
			}
		return infoSet;
		}
		

  // TODO think about how GUIKitUtils currently stores a single
  // cache in the whole lifetime of the VM when potentially
  // if these caches were stored per environment we could clear them
  // out when an environment shutsdown. But then if run again it would
  // have to recreate. Though this may still be better than currently
  // leaving the cache active for VM lifetime
	public static void flushCaches() {
    Collection vals = objectInfoSetCache.values();
    Iterator it = vals.iterator();
    while(it.hasNext()) {
      Object o = it.next();
      if (o != null) {
        ((GUIKitTypedObjectInfoSet)o).clear();
        }
      }
		objectInfoSetCache.clear();
		}

	public static void flushFromCaches(Class clz) {
		if (clz == null) {
			throw new NullPointerException();
			}
		Object o = objectInfoSetCache.remove(clz);
    if (o != null) {
      ((GUIKitTypedObjectInfoSet)o).clear();
      }
		}

  static GUIKitTypedObject createBean(GUIKitEnvironment env, GUIKitTypedObject bean, 
    URL url, String src, GUIKitTypedObject args[], Hashtable exposeHash, GUIKitLayoutInfo parentLayoutInfo)
      throws GUIKitException {

    if (src != null && src.startsWith("class:")) {
      try {
        // Latest BSF creates a Bean class already so no need for GUIKit to wrap a Bean with a Bean
        GUIKitTypedObject resultBean = null;
        
        String className = src.substring(6, src.length());
        
        if (className.startsWith("class:")) {
          resultBean = (GUIKitTypedObject)TypedObjectFactory.create(java.lang.Class.class, env.resolveClassName(className.substring(6)));
          }
        else {
          resultBean = (GUIKitTypedObject)MathematicaReflectionUtils.createBean(
            GUIKitEnvironment.getClassLoader(),
            className, 
            args);
          }
        // Is this the only aspect of the code below that is done on full documents
        // that we need to do when using a simple class: call?
        // What about setting up a new environment etc??
        // Or about setting up a default layout
        // TODO we need to decide what else needs done here for the class: case
        //if (resultBean != null &&  resultBean.getLayout() != null) {
        //  resultBean.getLayout().applyLayout(resultBean);
        //  }
        
        return resultBean;
        }
      catch(Exception exception) {
        Throwable throwable = (exception instanceof InvocationTargetException) ? ((InvocationTargetException)exception).getTargetException() : null;
        throw new GUIKitException(GUIKitException.REASON_CREATE_OBJECT_ERROR,
           "newInstance failed: " + exception +
           (throwable != null ? " target exception: " + throwable : ""),
             ((Throwable) (throwable != null ? throwable : ((Throwable) (exception)))));
        }
      }
    else {
      
      URL resolvedUrl = null;
      Document document = null;
      MathematicaBSFEngine mathEngine = env.getMathematicaEngine(true);
       
      // Here we just need any mathematica engine so we can use a parent's engine 
      resolvedUrl = MathematicaEngineUtils.getMathematicaURL(
          GUIKitEnvironment.RESOLVE_FUNCTION, url, src, mathEngine,
          env.getDriver().getURLCache());
      // I currently do not see a case where resolvedUrl needs to fall back to url
      // wihtout causing recursions
      //if (resolvedUrl == null && src == null) resolvedUrl = url;
      
      if (resolvedUrl == null) {
        throw new GUIKitException(GUIKitException.REASON_CREATE_OBJECT_ERROR,
          "cannot resolve resource " + src );
        }
  
      if (resolvedUrl.getPath().endsWith(".m")) {
        try {
          String xmlString = null;
          if ("file".equals(resolvedUrl.getProtocol())) {
            //System.out.println("File Get createBean: " + resolvedUrl.getPath());
            xmlString = mathEngine.requestExprFileToXMLString(
               resolvedUrl.getPath(), GUIKitUtils.GUI_XMLFORMAT);
            }
          else {
            String guiKitString = MathematicaEngineUtils.getContentAsString(resolvedUrl);
            xmlString = mathEngine.requestExprToXMLString(guiKitString, GUI_XMLFORMAT);
            }
          document = env.getDriver().parse(new StringReader(xmlString));
          }
        catch (Exception e) {
          throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, e.getMessage());
          }
        }
      // This should be .xml or do we check? Might support others or directories in future
      else {
        try {
          document = env.getDriver().parse(resolvedUrl);
          }
        catch (Exception e) {
          throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, e.getMessage());
          }
        }
  
      try {
        if (document == null) return null;
        
        GUIKitTypedObject resultBean = null;
        GUIKitEnvironment useEnv = new GUIKitEnvironment(env);

        // This scope should probably not be ACTION but either OBJECT or WIDGET 
        // as part of the args to the object or script
        registerAsScopeArguments(useEnv, args, MathematicaObjectRegistry.SCOPE_WIDGET);

        resultBean = useEnv.getDriver().processDocument(useEnv, document, bean, resolvedUrl, parentLayoutInfo);
      
        if (exposeHash != null) {
          Set keys = exposeHash.keySet();
          Iterator it = keys.iterator();
          while(it.hasNext()) {
            Object key = it.next();
            Object val = exposeHash.get(key);
            // Think about whether we wrap this in try/catch and allow null lookup without warnings
            Object inner = useEnv.lookupObject((String)key);
            if (inner != null) {
              env.registerObject( (String)val, inner);
              }
            }
          }

        return resultBean;
        }
      catch(Throwable throwable) {
        throw new GUIKitException(GUIKitException.REASON_CREATE_OBJECT_ERROR, "Cannot instantiate " + resolvedUrl + ": " + throwable, throwable);
        }      
        
      
      }

    }

  protected static GUIKitTypedObject getBeanField(GUIKitEnvironment env, GUIKitTypedObject bean, 
      String s, InvokeMode mode)
    throws Exception {
    Class class1 = bean.type != (java.lang.Class.class) ? bean.type : (Class)bean.value;
    // In order to match and find a field in a case-insensitive way we need to walk all fields
    Field[] fields = class1.getFields();
		Field field = null;
		String lowerName = s.toLowerCase();
    for (int i = 0; i < fields.length; ++i) {
    	if (lowerName.equalsIgnoreCase(fields[i].getName())) {
				field = fields[i];
				break;
    		}
    	}
    Class class2 = field.getType();
    Object obj = MathematicaMethodUtils.callGetField(field, bean.value, mode);
    return (GUIKitTypedObject)TypedObjectFactory.create(class2, obj);
    }
    
  protected static void setBeanField(GUIKitEnvironment env, GUIKitTypedObject bean, String s, GUIKitTypedObject bean1,
      InvokeMode mode) throws Exception {
		Class class1 = bean.type != (java.lang.Class.class) ? bean.type : (Class)bean.value;
		//	In order to match and find a field in a case-insensitive way we need to walk all fields
		Field[] fields = class1.getFields();
		Field field = null;
		String lowerName = s.toLowerCase();
		for (int i = 0; i < fields.length; ++i) {
			if (lowerName.equalsIgnoreCase(fields[i].getName())) {
				field = fields[i];
				break;
				}
			}
    
    Class class2 = field.getType();
    if(!class2.isAssignableFrom(bean1.type)) {
      bean1.value = MathematicaTypeConvertorRegistry.typeConvertorRegistry.convert(bean1, class2).value;
      }
    MathematicaMethodUtils.callSetField(field, bean.value, bean1.value, mode);
    }

  public static void resolveAndSetObjectProperty(Object handler, Object targetRef, String s,
    Integer integer, Object valObj, Object id, boolean checkForTargetString, 
    String invokeThread, String invokeWait) {
    InvokeMode mode = GUIKitUtils.determineInvokeMode(invokeThread, invokeWait);
    try {
    	GUIKitEnvironment env = resolveEnvironment(handler);
    	if (env != null) mode.setManager( env.getBSFManager());
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      setBeanProperty(env, useTarget, s, integer,
         (GUIKitTypedObject)TypedObjectFactory.create(valObj), mode);
      if (id != null && id instanceof String)
        resolveAndSetReference(handler, (String)id, valObj);
      }
    catch (Exception e) {
      handleException(handler, e);
      }
   }
   
    // In a move to convert properties and fields to members, we allow calls to property to fallback
    // to a check for field to combine them
  
    static void setBeanProperty(GUIKitEnvironment env, GUIKitTypedObject bean, String s, Integer integer, 
          GUIKitTypedObject bean1, InvokeMode mode)
        throws GUIKitException{
      if (bean == null || bean1 == null) return;
      
      try {
        // Either first or fallback, here is where we could lookup
        // in a PropertyRegistry a possible derived set property definition to use
        // For properties we ignore beaninfos which hide certain useful properties through reflection
        PropertyDescriptor propertydescriptor = getObjectInfoSet(bean).getCachedPropertyDescriptor(s);
        
        if (propertydescriptor == null) {
          throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, 
              "property '" + s + "' of object '" + bean.value + "' is not known");
          }
        Method method = null;
        Class class1 = null;
        if(integer != null) {
          if(!(propertydescriptor instanceof IndexedPropertyDescriptor))
             throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "attempt to set non-indexed property '" + s + "' as being indexed");
          IndexedPropertyDescriptor indexedpropertydescriptor = (IndexedPropertyDescriptor)propertydescriptor;
          method = indexedpropertydescriptor.getIndexedWriteMethod();
          class1 = indexedpropertydescriptor.getIndexedPropertyType();
          }
        else {
          method = propertydescriptor.getWriteMethod();
          class1 = propertydescriptor.getPropertyType();
          }
        if(method == null)
            throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "property '" + s + "' is not writable");
        Object convertedObj = null;
        try {
          convertedObj = MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(bean1, class1);
          }
        catch (MathematicaBSFException me){
          throw new GUIKitException(GUIKitException.REASON_UNKNOWN_TYPECVTOR, me.getMessage());
          }
  
        if(integer != null) {
          invokeMethod(method, bean.value, new Object[]{integer, convertedObj}, new Class[]{Integer.class, class1}, mode);
          } 
        else {
          invokeMethod(method, bean.value, new Object[]{convertedObj}, new Class[]{class1}, mode);
          }
        
        }
      catch (GUIKitException ie) {
        try {
          setBeanField(env, bean, s, bean1, mode);
          }
        catch (Exception ie2) {
          if (ie2 instanceof NoSuchFieldException) {
            throw ie;
            }
          else if (ie2 instanceof GUIKitException) {
            throw (GUIKitException)ie2;
            }
          else {
            throw ie;
            }
          }
      }
    }

  public static String[] resolveAndGetObjectMethodNames(Object handler, Object targetRef,  boolean checkForTargetString) {
    try {
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      return getBeanMethodNames( resolveEnvironment(handler), useTarget);
      }
    catch (Exception e) {
      handleException(handler, e);
      return null;  
      }
    }
  
  public static String[] getBeanMethodNames(GUIKitEnvironment env, GUIKitTypedObject bean) throws GUIKitException {
      if (bean == null) return null;
      Class thisClass = bean.type != (java.lang.Class.class) ? bean.type : (Class)bean.value;
      Method[] methods = thisClass.getMethods();
		  CapitalizedStringsVector nameList = new CapitalizedStringsVector(methods.length);
      for(int i = 0; i < methods.length; i++) {
				nameList.add(methods[i].getName());
        }
      return (String [])nameList.toArray(new String[]{});
      }

  public static Expr resolveAndGetObjectInvokeMethodPatterns(Object handler, Object targetRef, 
      boolean checkForTargetString, boolean verbosePattern) {
    try {
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      return getBeanInvokeMethodPatterns( resolveEnvironment(handler), useTarget, targetRef, 
        checkForTargetString, verbosePattern);
      }
    catch (Exception e) {
      handleException(handler, e);
      return null;  
      }
    }
  
  public static Expr getBeanInvokeMethodPatterns(GUIKitEnvironment env, GUIKitTypedObject bean,
        Object targetRef, boolean checkForTargetString, boolean verbosePattern) throws GUIKitException {
      if (bean == null) return null;
      
      LoopbackLink lnk = null;
      
      try {
        lnk = MathLinkFactory.createLoopbackLink();
        }
      catch (MathLinkException e) {}
      
      if (lnk == null) return null;
      
      int count = 0;
      
      Class thisClass = bean.type != (java.lang.Class.class) ? bean.type : (Class)bean.value;
      Method[] methods = thisClass.getMethods();
      
      count += methods.length;
      
      Expr resultExpr = null;
      
      try {
        lnk.putFunction("List", count);
        
        for(int i = 0; i < methods.length; i++) {
          Method m = methods[i];
          Class[] params = m.getParameterTypes();
          lnk.putFunction("InvokeMethod", 1 + params.length);
            lnk.putFunction("List", 2);
              if (checkForTargetString && (targetRef instanceof String)) {
                lnk.put((String)targetRef);
                }
              else {
                lnk.putFunction("Pattern", 2);
                  lnk.putSymbol("w");
                  lnk.putFunction("Blank", 0);
                  }
              lnk.put( capitalize(m.getName()) );
            for (int j=0; j < params.length; ++j) {
              writeClassPattern(lnk, params[j], verbosePattern);
              }
          }
          
        resultExpr = lnk.getExpr();
        }
      catch (MathLinkException me) {}
      
      return resultExpr;
      }
      
  public static String[] resolveAndGetObjectProperties(Object handler, Object targetRef, boolean checkForTargetString) {
    try {
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      return getBeanProperties( resolveEnvironment(handler), useTarget);
      }
    catch (Exception e) {
      handleException(handler, e);
      return null;  
      }
    }
  
  // We also now include field names since we are combining get and set of properties and fields
  public static String[] getBeanProperties(GUIKitEnvironment env, GUIKitTypedObject bean) throws GUIKitException {
      if (bean == null) return null;
		  CapitalizedStringsVector nameList = new CapitalizedStringsVector();
      
      // Either first or fallback, here is where we could lookup
      // in a PropertyRegistry a possible derived get property definition to add
      // Once added GUIKitEnvironment is required for a specific object registry call

      BeanInfo beaninfo = getObjectInfoSet(bean).getBeanInfo();
      PropertyDescriptor propDescriptors[] = beaninfo.getPropertyDescriptors();
      for(int i = 0; i < propDescriptors.length; i++) {
        nameList.add( propDescriptors[i].getName());
        }
        
      // TODO we may want to allow or default a filter for access level
      Class class1 = bean.type != (java.lang.Class.class) ? bean.type : (Class)bean.value;
      
      // TODO this shows public and private fields but not superclass fields
      // we may need to change this or use the common utils package
      Field[] fields = class1.getDeclaredFields();
      for(int i = 0; i < fields.length; i++) {
        if (Modifier.isPublic( fields[i].getModifiers()))
          nameList.add( fields[i].getName());
        }
        
      return (String [])nameList.toArray(new String[]{});
      }

  private static void writeClassPattern(MathLink lnk, Class c, boolean verbosePattern) throws MathLinkException {
    if (c == null) 
      lnk.putFunction("Blank", 0);
    else if (c == String.class) {
      lnk.putFunction("Blank", 1);
        lnk.putSymbol("String");
      }
    else if (c == byte.class || c == Byte.class || 
             c == int.class || c == Integer.class || c == BigInteger.class) {
      lnk.putFunction("Blank", 1);
        lnk.putSymbol("Integer");
      }
    else if (c == float.class || c == Float.class || 
             c == double.class || c == Double.class  || c == BigDecimal.class) {
      lnk.putFunction("Blank", 1);
        lnk.putSymbol("Real");
      }
    else if (c == boolean.class || c == Boolean.class) {
      if (verbosePattern) {
        lnk.putFunction("PatternTest", 2);
          lnk.putFunction("Blank", 0);
          lnk.putFunction("Function", 1);
            lnk.putFunction("MatchQ", 2);
              lnk.putFunction("Slot", 1);
                lnk.put(1);
              lnk.putFunction("Alternatives", 2);
                lnk.putSymbol("True");
                lnk.putSymbol("False");
        }
      else {
        lnk.putFunction("Pattern", 2);
          lnk.putSymbol("bool");
          lnk.putFunction("Blank", 0);
        }
      }
    else if (c == String[].class) {
      lnk.putFunction("List", 1);
        lnk.putFunction("BlankNullSequence", 1);
          lnk.putSymbol("String");
      }
    else if (c == byte[].class || c == Byte[].class ||
             c == int[].class || c == Integer[].class || c == BigInteger[].class) {
      lnk.putFunction("List", 1);
        lnk.putFunction("BlankNullSequence", 1);
          lnk.putSymbol("Integer");
      }
    else if (c == float[].class || c == Float[].class || 
             c == double[].class || c == Double[].class  || c == BigDecimal[].class) {
      lnk.putFunction("List", 1);
        lnk.putFunction("BlankNullSequence", 1);
          lnk.putSymbol("Real");
      }
    else if (c == boolean[].class || c == Boolean[].class) {
      if (verbosePattern) {
        lnk.putFunction("List", 1);
          lnk.putFunction("Pattern", 2);
            lnk.putSymbol("bool");
            lnk.putFunction("BlankNullSequence", 0);
        }
      else {
        lnk.putFunction("List", 1);
          lnk.putFunction("Pattern", 2);
            lnk.putSymbol("bool");
            lnk.putFunction("BlankNullSequence", 0);
        }
      }
    else if (c == char.class || c == Character.class ||
             c == short.class || c == Short.class || c == long.class || c == Long.class) {
      lnk.putFunction("Blank", 1);
        lnk.putSymbol("Integer");
      }
    else if (c == char[].class || c == Character[].class || 
             c == short[].class || c == Short[].class || c == long[].class || c == Long[].class) {
      lnk.putFunction("List", 1);
        lnk.putFunction("BlankNullSequence", 1);
          lnk.putSymbol("Integer");
      }
    // TODO think about other Widget types that we can return patterns for
    /* Possible additional:
     *   maybe Icon or Image, maybe Wizard classes, maybe Action
     *   think about pref to use a WidgetReference["ref"] choice as well
     */
    else if (c == Dimension.class) {
      lnk.putFunction("Widget", 2);
        lnk.put("Dimension");
        lnk.putFunction("List", 2);
          lnk.putFunction("Rule", 2);
            lnk.put("Width");
            writeClassPattern(lnk, int.class, verbosePattern);
          lnk.putFunction("Rule", 2);
            lnk.put("Height");
            writeClassPattern(lnk, int.class, verbosePattern);
      }
    else if (c == Expr.class) {
      lnk.putFunction("Pattern", 2);
        lnk.putSymbol("expr");
        lnk.putFunction("Blank", 0);
      }
    else if (c == Point.class) {
      lnk.putFunction("Widget", 2);
        lnk.put("Point");
        lnk.putFunction("Rule", 2);
          lnk.putSymbol("InitialArguments");
          lnk.putFunction("List", 2);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("x");
              writeClassPattern(lnk, int.class, verbosePattern);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("y");
              writeClassPattern(lnk, int.class, verbosePattern);
      }
    else if (c == Color.class) {
      lnk.putFunction("Widget", 2);
        lnk.put("Color");
        lnk.putFunction("Rule", 2);
          lnk.putSymbol("InitialArguments");
          lnk.putFunction("List", 3);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("r");
              writeClassPattern(lnk, int.class, verbosePattern);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("g");
              writeClassPattern(lnk, int.class, verbosePattern);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("b");
              writeClassPattern(lnk, int.class, verbosePattern);
      }
    else if (c == Rectangle.class) {
      lnk.putFunction("Widget", 2);
        lnk.put("Rectangle");
        lnk.putFunction("Rule", 2);
          lnk.putSymbol("InitialArguments");
          lnk.putFunction("List", 4);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("x");
              writeClassPattern(lnk, int.class, verbosePattern);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("y");
              writeClassPattern(lnk, int.class, verbosePattern);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("w");
              writeClassPattern(lnk, int.class, verbosePattern);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("h");
              writeClassPattern(lnk, int.class, verbosePattern);
      }
    else if (c == Font.class) {
      lnk.putFunction("Widget", 2);
        lnk.put("Font");
        lnk.putFunction("Rule", 2);
          lnk.putSymbol("InitialArguments");
          lnk.putFunction("List", 3);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("name");
              writeClassPattern(lnk, String.class, verbosePattern);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("style");
              writeClassPattern(lnk, int.class, verbosePattern);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("size");
              writeClassPattern(lnk, int.class, verbosePattern);
      }
    else if (c == KeyStroke.class) {
      lnk.putFunction("Widget", 2);
        lnk.put("KeyStroke");
        lnk.putFunction("Rule", 2);
          lnk.putSymbol("InitialArguments");
          lnk.putFunction("List", 1);
            lnk.putFunction("Pattern", 2);
              lnk.putSymbol("s");
              writeClassPattern(lnk, String.class, verbosePattern);
      }
    else if (c.isArray()) {
      lnk.putFunction("List", 1);
        lnk.putFunction("BlankNullSequence", 0);
      }
    else {
      if (verbosePattern) {
        lnk.putFunction("PatternTest", 2);
          lnk.putFunction("Blank", 0);
          lnk.putFunction("Function", 1);
            lnk.putFunction("InstanceOf", 2);
              lnk.putFunction("Slot", 1);
                lnk.put(1);
              lnk.putFunction("LoadJavaClass", 1);
                lnk.put(c.getName());
        }
      else lnk.putFunction("Blank", 0);
      }
    }
  
  public static Expr resolveAndGetObjectSetPropertyValuePatterns(Object handler, Object targetRef, 
      boolean checkForTargetString, boolean verbosePattern) {
    try {
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      return getBeanSetPropertyValuePatterns( resolveEnvironment(handler), useTarget, targetRef, 
        checkForTargetString, verbosePattern);
      }
    catch (Exception e) {
      handleException(handler, e);
      return null;  
      }
    }
  
  // We also now include field names since we are combining get and set of properties and fields
  public static Expr getBeanSetPropertyValuePatterns(GUIKitEnvironment env, GUIKitTypedObject bean,
         Object targetRef, boolean checkForTargetString, boolean verbosePattern) throws GUIKitException {
      if (bean == null) return null;
      LoopbackLink lnk = null;
      try {
        lnk = MathLinkFactory.createLoopbackLink();
        }
      catch (MathLinkException e) {}
      if (lnk == null) return null;
      int count = 0;
      
      // Either first or fallback, here is where we could lookup
      // in a PropertyRegistry a possible derived get property definition to add
      // Once added GUIKitEnvironment is required for a specific object registry call

      BeanInfo beaninfo = getObjectInfoSet(bean).getBeanInfo();
      PropertyDescriptor propDescriptors[] = beaninfo.getPropertyDescriptors();
       // TODO we may want to allow or default a filter for access level
      Class class1 = bean.type != (java.lang.Class.class) ? bean.type : (Class)bean.value;
      // TODO this shows public and private fields but not superclass fields
      // we may need to change this or use the common utils package
      Field[] fields = class1.getDeclaredFields();
      for(int i = 0; i < fields.length; i++) {
        if (Modifier.isPublic( fields[i].getModifiers()))
          ++count;
        }
      for(int i = 0; i < propDescriptors.length; i++) {
        if (propDescriptors[i] instanceof IndexedPropertyDescriptor) {
          if (((IndexedPropertyDescriptor)propDescriptors[i]).getIndexedWriteMethod() != null) ++count;
          }
        else {
          if (propDescriptors[i].getWriteMethod() != null) ++count;
          }
        }
      
      Expr resultExpr = null;
      
      try {
        lnk.putFunction("List", count);
        
        for(int i = 0; i < propDescriptors.length; i++) {
          if (propDescriptors[i] instanceof IndexedPropertyDescriptor) {
            if (((IndexedPropertyDescriptor)propDescriptors[i]).getIndexedWriteMethod() == null) continue;
            lnk.putFunction("SetPropertyValue", 2);
            lnk.putFunction("List", 3);
              if (checkForTargetString && (targetRef instanceof String)) {
                lnk.put((String)targetRef);
                }
              else {
                lnk.putFunction("Pattern", 2);
                  lnk.putSymbol("w");
                  lnk.putFunction("Blank", 0);
                }
              lnk.put( capitalize(propDescriptors[i].getName()));
              lnk.putFunction("Blank", 1);
                lnk.putSymbol("Integer");
            writeClassPattern(lnk, 
              ((IndexedPropertyDescriptor)propDescriptors[i]).getIndexedPropertyType(), 
              verbosePattern);
            }
          else {
            if (propDescriptors[i].getWriteMethod() == null) continue;
            lnk.putFunction("SetPropertyValue", 2);
              lnk.putFunction("List", 2);
                if (checkForTargetString && (targetRef instanceof String)) {
                  lnk.put((String)targetRef);
                  }
                else {
                  lnk.putFunction("Pattern", 2);
                    lnk.putSymbol("w");
                    lnk.putFunction("Blank", 0);
                  }
                lnk.put( capitalize(propDescriptors[i].getName()));
              writeClassPattern(lnk, propDescriptors[i].getPropertyType(),
                verbosePattern);
            }
          }
        for(int i = 0; i < fields.length; i++) {
          if (Modifier.isPublic( fields[i].getModifiers())) {
            lnk.putFunction("SetPropertyValue", 2);
              lnk.putFunction("List", 2);
                if (checkForTargetString && (targetRef instanceof String)) {
                  lnk.put((String)targetRef);
                  }
                else {
                  lnk.putFunction("Pattern", 2);
                    lnk.putSymbol("w");
                    lnk.putFunction("Blank", 0);
                  }
                lnk.put( capitalize(fields[i].getName()));
              writeClassPattern(lnk, fields[i].getType(), verbosePattern);
            }
          }
        resultExpr = lnk.getExpr();
        }
      catch (MathLinkException me) {}
      
      return resultExpr;
      }
      
  public static Expr resolveAndGetObjectPropertyValuePatterns(Object handler, Object targetRef,
      boolean checkForTargetString, boolean verbosePattern) {
    try {
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      return getBeanPropertyValuePatterns( resolveEnvironment(handler), useTarget, targetRef, 
        checkForTargetString, verbosePattern);
      }
    catch (Exception e) {
      handleException(handler, e);
      return null;  
      }
    }
  
  // We also now include field names since we are combining get and set of properties and fields
  public static Expr getBeanPropertyValuePatterns(GUIKitEnvironment env, GUIKitTypedObject bean,
         Object targetRef, boolean checkForTargetString, boolean verbosePattern) throws GUIKitException {
      if (bean == null) return null;
      LoopbackLink lnk = null;
      try {
        lnk = MathLinkFactory.createLoopbackLink();
        }
      catch (MathLinkException e) {}
      if (lnk == null) return null;
      int count = 0;
      
      // Either first or fallback, here is where we could lookup
      // in a PropertyRegistry a possible derived get property definition to add
      // Once added GUIKitEnvironment is required for a specific object registry call

      BeanInfo beaninfo = getObjectInfoSet(bean).getBeanInfo();
      PropertyDescriptor propDescriptors[] = beaninfo.getPropertyDescriptors();
       // TODO we may want to allow or default a filter for access level
      Class class1 = bean.type != (java.lang.Class.class) ? bean.type : (Class)bean.value;
      // TODO this shows public and private fields but not superclass fields
      // we may need to change this or use the common utils package
      Field[] fields = class1.getDeclaredFields();
      for(int i = 0; i < fields.length; i++) {
        if (Modifier.isPublic( fields[i].getModifiers()))
          ++count;
        }
      for(int i = 0; i < propDescriptors.length; i++) {
        if (propDescriptors[i] instanceof IndexedPropertyDescriptor) {
          if (((IndexedPropertyDescriptor)propDescriptors[i]).getIndexedReadMethod() != null) ++count;
          }
        else {
          if (propDescriptors[i].getReadMethod() != null) ++count;
          }
        }
      
      Expr resultExpr = null;
      
      try {
        lnk.putFunction("List", count);
        
        for(int i = 0; i < propDescriptors.length; i++) {
          if (propDescriptors[i] instanceof IndexedPropertyDescriptor) {
            if (((IndexedPropertyDescriptor)propDescriptors[i]).getIndexedReadMethod() == null) continue;
            lnk.putFunction("PropertyValue", 1);
            lnk.putFunction("List", 3);
              if (checkForTargetString && (targetRef instanceof String)) {
                lnk.put((String)targetRef);
                }
              else {
                lnk.putFunction("Pattern", 2);
                  lnk.putSymbol("w");
                  lnk.putFunction("Blank", 0);
                }
              lnk.put( capitalize(propDescriptors[i].getName()));
              lnk.putFunction("Blank", 1);
                lnk.putSymbol("Integer");
            }
          else {
            if (propDescriptors[i].getReadMethod() == null) continue;
            lnk.putFunction("PropertyValue", 1);
              lnk.putFunction("List", 2);
                if (checkForTargetString && (targetRef instanceof String)) {
                  lnk.put((String)targetRef);
                  }
                else {
                  lnk.putFunction("Pattern", 2);
                    lnk.putSymbol("w");
                    lnk.putFunction("Blank", 0);
                  }
                lnk.put( capitalize(propDescriptors[i].getName()));
            }
          }
        for(int i = 0; i < fields.length; i++) {
          if (Modifier.isPublic( fields[i].getModifiers())) {
            lnk.putFunction("PropertyValue", 2);
              lnk.putFunction("List", 2);
                if (checkForTargetString && (targetRef instanceof String)) {
                  lnk.put((String)targetRef);
                  }
                else {
                  lnk.putFunction("Pattern", 2);
                    lnk.putSymbol("w");
                    lnk.putFunction("Blank", 0);
                  }
                lnk.put( capitalize(fields[i].getName()));
            }
          }
        resultExpr = lnk.getExpr();
        }
      catch (MathLinkException me) {}
      
      return resultExpr;
      }
      
  public static String[] resolveAndGetWidgetNames(Object handler) {
    try {
      GUIKitEnvironment env = resolveEnvironment(handler);
      if (env != null) return env.getReferenceNames(true);
      else return new String[]{};
      }
    catch (Exception e) {
      handleException(handler, e);
      return new String[]{};  
      }
    }
    
  public static void resolveAndSetReference(Object handler, Object id, Object obj) {
    resolveAndSetReference(handler, id, obj, MathematicaObjectRegistry.SCOPE_OBJECT);
    }

  public static void resolveAndSetReference(Object handler, Object id, Object obj, int scope) {
    if (id == null || !(id instanceof String)) return;
    try {
      if (handler instanceof MathematicaBSFFunctions) {
        ((MathematicaBSFFunctions)handler).registerBean((String)id, obj, scope);
        }
      else if (handler instanceof GUIKitDriver) {
        ((GUIKitDriver)handler).registerObject((String)id, obj, scope);
        }
      else {
        // TODO really should throw exception if passed invalid handler
        }
      }
    catch (Exception e) {
      handleException(handler, e);
      }
    }
    
   public static void resolveAndUnsetReference(Object handler, Object id) {
     resolveAndUnsetReference(handler, id,  MathematicaObjectRegistry.SCOPE_OBJECT);
     }
   
   public static void resolveAndUnsetReference(Object handler, Object id, int scope) {
    if (id == null || !(id instanceof String)) return;
    try {
      if (handler instanceof MathematicaBSFFunctions) {
        ((MathematicaBSFFunctions)handler).unregisterBean((String)id, scope);
        }
      else if (handler instanceof GUIKitDriver) {
        ((GUIKitDriver)handler).unregisterObject((String)id, scope);
        }
      else {
        // TODO really should throw exception if passed invalid handler
        }
      }
    catch (Exception e) {
      handleException(handler, e);
      }
    }
    
   public static Object resolveAndGetReference(Object handler, Object id) {
     return resolveAndGetReference(handler, id, 
       MathematicaObjectRegistry.SCOPE_FIRST, MathematicaObjectRegistry.SCOPE_LAST);
     }
   public static Object resolveAndGetReference(Object handler, Object id, int maxScope) {
     return resolveAndGetReference(handler, id, 
       MathematicaObjectRegistry.SCOPE_FIRST, maxScope);
     }
     
   public static Object resolveAndGetReference(Object handler, Object id, int minScope, int maxScope) {
    Object result = null;
    if (id == null || !(id instanceof String)) return result;
    try {
      if (handler instanceof MathematicaBSFFunctions) {
        result = ((MathematicaBSFFunctions)handler).lookupBean((String)id, minScope, maxScope);
        }
      else if (handler instanceof GUIKitDriver) {
        result = ((GUIKitDriver)handler).lookupObject((String)id, minScope, maxScope);
        }
      else {
        result = null; // TODO really should throw exception if passed invalid handler
        }
      }
    catch (Exception e) {
      handleException(handler, e);
      }
    return result;
    }
    
  public static String[] resolveAndGetObjectEvents(Object handler, Object targetRef, boolean checkForTargetString) {
    try {
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      return getBeanEvents( resolveEnvironment(handler), useTarget);
      }
    catch (Exception e) {
      handleException(handler, e);
      return null;  
      }
    }
  
  public static void resolveAndCloseGUIObject(Object handler) {
    try {
      if (handler instanceof MathematicaBSFFunctions) {
        Object driver = ((MathematicaBSFFunctions)handler).lookupBean( MathematicaBSFEngine.ID_DRIVER);
        Object executeObject = ((MathematicaBSFFunctions)handler).lookupBean( GUIKitDriver.ID_ROOTOBJECT);
        if (driver != null && executeObject != null && driver instanceof GUIKitDriver) {
           ((GUIKitDriver)driver).requestClose(executeObject);
           }
        }
      else if (handler instanceof GUIKitDriver) {
        Object executeObject = ((GUIKitDriver)handler).lookupObject( GUIKitDriver.ID_ROOTOBJECT);
        ((GUIKitDriver)handler).requestClose(executeObject);
        }
      else {
        // TODO really should throw exception if passed invalid handler
        }
      }
    catch (Exception e) {
      handleException(handler, e);
      }
    }
    
  public static String[] getBeanEvents(GUIKitEnvironment env, GUIKitTypedObject bean) throws GUIKitException {
    if (bean == null) return null;
		CapitalizedStringsVector nameList = new CapitalizedStringsVector();
    
    BeanInfo beaninfo = getObjectInfoSet(bean).getBeanInfo();
    EventSetDescriptor afeaturedescriptor[] = beaninfo.getEventSetDescriptors();
    
    for(int i = 0; i < afeaturedescriptor.length; i++) {
      nameList.add(afeaturedescriptor[i].getName());
      Method[] methods = afeaturedescriptor[i].getListenerMethods();
      for (int j = 0; j < methods.length; j++) {
        nameList.add(methods[j].getName());
        }
      }

    return (String [])nameList.toArray(new String[]{});
    }
      
  public static Object resolveAndGetObjectProperty(Object handler, Object targetRef, 
        String s, Integer integer, Object id, boolean checkForTargetString, 
        String invokeThread, String invokeWait) {
    InvokeMode mode = GUIKitUtils.determineInvokeMode(invokeThread, invokeWait);
    try {
      Object result = null;
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      
      GUIKitEnvironment env = resolveEnvironment(handler);
      if (env != null) mode.setManager( env.getBSFManager());
  
      GUIKitTypedObject resultBean = getBeanProperty( resolveEnvironment(handler), useTarget, s, integer, mode);
      result = (resultBean != null ? resultBean.value : null);
      if (id != null && id instanceof String) resolveAndSetReference(handler, (String)id, result);
      return result;
      }
    catch (Exception e) {
      handleException(handler, e);
      return null;  
      }
          
    }
  
    // In a move to convert properties and fields to members, we allow calls to property to fallback
    // to a check for field to combine them
    public static GUIKitTypedObject getBeanProperty(GUIKitEnvironment env, GUIKitTypedObject bean, 
        String s, Integer integer, InvokeMode mode) throws GUIKitException {
      GUIKitTypedObject result = null;
      
      if (bean == null || bean.value == null) return null;
      try {
        // Either first or fallback, here is where we could lookup
        // in a PropertyRegistry a possible derived get property definition to use
        PropertyDescriptor propertydescriptor = getObjectInfoSet(bean).getCachedPropertyDescriptor(s);
        if (propertydescriptor == null) {
          throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, 
              "property '" + s + "' of object '" + bean.value + "' is not known");
          }
        
        Method method;
        Class class1;
        if (integer != null) {
          if(!(propertydescriptor instanceof IndexedPropertyDescriptor))
            throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "attempt to get non-indexed property '" + s + "' as being indexed");
          IndexedPropertyDescriptor indexedpropertydescriptor = (IndexedPropertyDescriptor)propertydescriptor;
          method = indexedpropertydescriptor.getIndexedReadMethod();
          class1 = indexedpropertydescriptor.getIndexedPropertyType();
          }
        else {
          method = propertydescriptor.getReadMethod();
          class1 = propertydescriptor.getPropertyType();
          }
        if(method == null)
            throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "property '" + s + "' is not readable");
            
        result = (GUIKitTypedObject)TypedObjectFactory.create(class1, 
        	invokeMethod(method, bean.value, 
            integer == null ? null : (new Object[] {integer}), 
            integer == null ? null : (new Class[] {Integer.class}),
            mode) 
        	);  
      }
      catch (GUIKitException ie) {
        try {
          result = getBeanField(env, bean, s, mode);
          }
        catch (Exception ie2) {
          if (ie2 instanceof NoSuchFieldException) {
            throw ie;
            }
          else if (ie2 instanceof GUIKitException) {
            throw (GUIKitException)ie2;
            }
          else {
            throw ie;
            }
          }
        }
      return result;
    }

  static GUIKitTypedObject addEventListener(GUIKitEnvironment env, GUIKitTypedObject bean,
          String eventName, String eventFilter, GUIKitTypedObject bean1, InvokeMode mode)
        throws GUIKitException {
      // consider throwing exception here
      if (bean == null || bean.value == null) return null;

      BeanInfo beaninfo = getObjectInfoSet(bean).getBeanInfo();
      EventSetDescriptor eventsetdescriptor = null;
      EventSetDescriptor afeaturedescriptor[] = beaninfo.getEventSetDescriptors();

      for(int i = 0; i < afeaturedescriptor.length; i++) {
        if(eventName.equalsIgnoreCase(afeaturedescriptor[i].getName())) {
          eventsetdescriptor = afeaturedescriptor[i];
          break;
          }
        }

      // Secondary check is to use name as a listener method (ie filter mode)
      if (eventsetdescriptor == null) {
        for(int i = 0; i < afeaturedescriptor.length; i++) {
          Method[] methods = afeaturedescriptor[i].getListenerMethods();
          for (int j = 0; j < methods.length; j++) {
            if(eventName.equalsIgnoreCase(methods[j].getName())) {
              eventsetdescriptor = afeaturedescriptor[i];
              break;
              }
            }
          }
        }

      if (eventsetdescriptor == null) {
        throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT,
          "event'" + eventName + "' of widget '" + bean.value + "' is not known");
        }

      // Do we really care of a filter exists??
      if(eventFilter != null)
          throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT,
            "event filter '" + eventFilter + "' not supported via simple event binding " +
            "API in JDK1.1.* - only for property/vetochange " + "events in 1.2");

      Class class1 = eventsetdescriptor.getListenerType();
      if(!class1.isInstance(bean1.value)) {
        throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT,
          "target widget '" + bean1.value + "' in event binding does not implement '" + class1 + "'");
        }
      else {
        invokeMethod(eventsetdescriptor.getAddListenerMethod(), bean.value, 
          new Object[] {bean1.value}, new Class[] {bean1.type}, mode);
        return bean1;
        }
      }

  // Now supports just using an eventName that works like a filter
  // TODO, would be nice to break this out to an Object version independent of XML elements

  static GUIKitTypedObject bindEventToElement(GUIKitEnvironment env, GUIKitTypedObject bean,
          String eventName, String eventFilter, Element element, GUIKitTypedObject bean1, URL url,
           InvokeMode mode)
      throws GUIKitException {

      String useEventFilter = eventFilter;

      // consider throwing exception here
      if (bean == null || bean.value == null) return null;

      BeanInfo beaninfo = getObjectInfoSet(bean).getBeanInfo();
      EventSetDescriptor eventsetdescriptor = null;
      EventSetDescriptor afeaturedescriptor[] = beaninfo.getEventSetDescriptors();

      for(int i = 0; i < afeaturedescriptor.length; i++) {
        if(eventName.equalsIgnoreCase(afeaturedescriptor[i].getName())) {
          eventsetdescriptor = afeaturedescriptor[i];
          break;
          }
        }

      // Secondary check is to use name as a listener method (ie filter mode)
      if (eventsetdescriptor == null) {
        for(int i = 0; i < afeaturedescriptor.length; i++) {
          Method[] methods = afeaturedescriptor[i].getListenerMethods();
          for (int j = 0; j < methods.length; j++) {
            if(eventName.equalsIgnoreCase(methods[j].getName())) {
              eventsetdescriptor = afeaturedescriptor[i];
              useEventFilter = methods[j].getName();
              break;
              }
            }
          }
        }

      if (eventsetdescriptor == null) {
        throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT,
          "event'" + eventName + "' of widget '" + bean.value + "' is not known");
        }

      Class listenerClass = eventsetdescriptor.getListenerType();

      if(!eventName.equalsIgnoreCase(MathematicaEngineUtils.ATTVAL_PROPERTY_CHANGE) &&
         !eventName.equalsIgnoreCase(MathematicaEngineUtils.ATTVAL_VETOABLE_CHANGE)) {
         	
        Class aclass[] = GUIKitUtils.getCommonArgTypes(listenerClass, useEventFilter);
        
        if(aclass == null) {
          String exceptionMessage;
          if(useEventFilter == null)
            exceptionMessage = "If a filter is not specified, the argument list of each and every method in the Listener interface must be identical. Some methods in " +
              listenerClass + " have argument lists that are not identical.";
          else
            exceptionMessage = "Either a method with the name " + useEventFilter +
              " was not found in " + listenerClass + ", or there " + "exists more than one method in " +
              listenerClass + " with the name " + useEventFilter + ".";
          throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, exceptionMessage);
          }
        }

      GUIKitEventProcessor eventprocessor = null;
      
      if (element.getTagName().equals(GUIKitUtils.ELEM_SCRIPT)) {
        eventprocessor = new GUIKitEventScriptProcessor();
        }
      else {
        eventprocessor = new GUIKitEventProcessor();
        }
      
      eventprocessor.setFilter(useEventFilter);
      eventprocessor.setContext(bean1);
      eventprocessor.setContextURL(url);
      eventprocessor.setGUIKitEnvironment(env);
      eventprocessor.setInvokeMode(mode);
      eventprocessor.setRootElement(element);
      
      GUIKitTypedObject adapter = null;
      try {
        adapter = (GUIKitTypedObject)MathematicaEventAdapterProxy.newInstance(
           env.getMathematicaEngine(true).getClassLoader(),
           eventsetdescriptor, eventprocessor, bean.value);
        }
      catch (MathematicaBSFException me) {
        throw new GUIKitException(GUIKitException.REASON_CREATE_OBJECT_ERROR, me.getMessage());
        }
        
      return adapter;
      }
      
  static GUIKitTypedObject registerEndModalScript(GUIKitEnvironment env, GUIKitTypedObject bean, Element element, 
          GUIKitTypedObject bean1, URL url, InvokeMode mode)
        throws GUIKitException {

      GUIKitEventScriptProcessor eventprocessor = new GUIKitEventScriptProcessor();
      eventprocessor.setContext(bean1);
      eventprocessor.setContextURL(url);
      eventprocessor.setGUIKitEnvironment(env);
      eventprocessor.setInvokeMode(mode);
      eventprocessor.setRootElement(element);
      
      env.registerObject(MathematicaBSFEngine.ID_ENDMODALSCRIPT, eventprocessor,
        MathematicaObjectRegistry.SCOPE_OBJECT);
      return (GUIKitTypedObject)TypedObjectFactory.create(eventprocessor);
      }

  private static void handleException(Object handler, Exception e) {
    if (handler instanceof MathematicaBSFFunctions) {
      ((MathematicaBSFFunctions)handler).handleException(e);
      }
    else if (handler instanceof GUIKitDriver) {
      ((GUIKitDriver)handler).handleException(e);
      } 
  }
  
  private static GUIKitTypedObject resolveTarget(Object handler, Object targetRef, boolean checkForTargetString) throws GUIKitException {
    if (targetRef == null) return null;
    
    GUIKitTypedObject useTarget = null;
    
    if (checkForTargetString && (targetRef instanceof String)) {
      if(((String)targetRef).startsWith("class:")) {
        useTarget = (GUIKitTypedObject)TypedObjectFactory.create(java.lang.Class.class, 
            resolveEnvironment(handler).resolveClassName(((String)targetRef).substring(6)));
        }
      else {
        Object targetObject = null;
        if (handler instanceof MathematicaBSFFunctions) {
          targetObject = ((MathematicaBSFFunctions)handler).lookupBean((String)targetRef);
          }
        else if (handler instanceof GUIKitDriver) {
          targetObject = ((GUIKitDriver)handler).lookupObject((String)targetRef);
          }
        if (targetObject != null) {
          useTarget = (GUIKitTypedObject)TypedObjectFactory.create(targetObject);
          }
        }
      }
    else {
    	if (targetRef instanceof GUIKitTypedObject) useTarget = (GUIKitTypedObject)targetRef;
			else useTarget  = (GUIKitTypedObject)TypedObjectFactory.create(targetRef);
    	}
    
    if (useTarget == null) throw 
      new GUIKitException(GUIKitException.REASON_UNKNOWN_OBJECT, "target " + targetRef.toString() + " was not resolved to a valid object instance");
      
    return useTarget;
    }

  private static GUIKitEnvironment resolveEnvironment(Object handler) {
    if (handler instanceof MathematicaBSFFunctions) {
      Object obj = ((MathematicaBSFFunctions)handler).lookupBean(GUIKitDriver.ID_GUIKITENV);
      if (obj != null && obj instanceof GUIKitEnvironment)
        return (GUIKitEnvironment)obj;
      else return null;
      }
    else if (handler instanceof GUIKitDriver) {
      return ((GUIKitDriver)handler).getGUIKitEnvironment();
      }
    else {
      // TODO really should throw exception here
      }
    return null;
    }
    
  public static InvokeMode determineInvokeMode(String invokeThread, String invokeWait) {
    String threadContext = InvokeMode.THREAD_CURRENT;
    int executionMode = InvokeMode.EXECUTE_LATER;
    
    if (invokeThread != null) {
      if (invokeThread.equalsIgnoreCase("Dispatch")) threadContext = InvokeMode.THREAD_DISPATCH;
      else if (invokeThread.equalsIgnoreCase("New")) threadContext = InvokeMode.THREAD_NEW;
      else if (invokeThread.equalsIgnoreCase("Current")) {
        threadContext = InvokeMode.THREAD_CURRENT;
        executionMode = InvokeMode.EXECUTE_WAIT;
        }
      else threadContext = invokeThread;
      }
    if (invokeWait != null && 
        (invokeWait.equalsIgnoreCase("true") || invokeWait.equalsIgnoreCase("yes"))) {
      executionMode = InvokeMode.EXECUTE_WAIT;
      }
    
    return new InvokeMode(threadContext, executionMode);
    }
   
  public static Object resolveAndInvokeObjectMethodName(Object handler, String methodName, Object targetRef, 
    Vector aobjvector, Object id, boolean checkForTargetString, String invokeThread, String invokeWait) {
    InvokeMode mode = determineInvokeMode(invokeThread, invokeWait);
    try {
      GUIKitTypedObject useTarget = resolveTarget(handler, targetRef, checkForTargetString);
      if (useTarget == null || useTarget.value == null) return null;
      
			GUIKitEnvironment env = resolveEnvironment(handler);
			if (env != null) mode.setManager(env.getBSFManager());
			
      Object result = null;
      if (aobjvector != null) {
        Object aobj[] = null;
        aobj = new Object[aobjvector.size()];
        for(int i = 0; i < aobjvector.size(); i++) {
          aobj[i] = aobjvector.elementAt(i);
          }
        result = MathematicaMethodUtils.invokeMethod(useTarget, methodName, aobj, mode);
        }
      else {
        result = MathematicaMethodUtils.invokeMethod(useTarget, methodName, mode);
        }
      if (id != null && id instanceof String)
        resolveAndSetReference(handler, (String)id, result);
      return result;
      }
    catch (Exception exception) {
      Throwable throwable = (exception instanceof InvocationTargetException) ? 
        ((InvocationTargetException)exception).getTargetException() : null;
      handleException(handler, 
        new GUIKitException(GUIKitException.REASON_CALL_METHOD_ERROR,
              "method invocation failed when calling : " + methodName + " : " + 
              exception + (throwable != null ? " target exception: " + throwable : ""), throwable));
      return null;  
      }
          
    }

  static Object invokeMethod(Method method, Object obj, Object aobj[], Class[] argTypes, InvokeMode mode) 
      throws GUIKitException {
    try {
      return MathematicaMethodUtils.callMethod(method, obj, aobj, argTypes, mode);
      }
    catch(Exception exception) {
      Throwable throwable = (exception instanceof InvocationTargetException) ? ((InvocationTargetException)exception).getTargetException() : null;
      throw new GUIKitException(GUIKitException.REASON_CALL_METHOD_ERROR, "method invocation failed when calling : " + method.getName() + " : " + exception + (throwable != null ? " target exception: " + throwable : ""), throwable);
      }
    }


   // TODO, if we stick with "##" consider supporting a check for "##n" on registry lookup
   // to return a subarray of the specified elements
        
  // TODO, might be useful to register #n class type too since we have them??
  
  public static void registerAsScopeArguments(GUIKitEnvironment env, GUIKitTypedObject[] args, int scope) {
    if(args != null) {
      GUIKitTypedObject obj = null;
      Object[] objArr = new Object[args.length];
      if (args.length > 0) {
        obj = args[0];
        if (obj != null && obj.value != null)
          env.registerObject("#", obj.value, scope);
        else env.unregisterObject("#", scope);
        for(int i = 0; i < args.length; i++) {
          obj = args[i];
          if (obj == null || obj.value == null) {
            env.unregisterObject("#" + (i + 1), scope);
            continue;
            }
          objArr[i] = obj.value;
          // index is 0-based but Mathematica uses 1-based
          env.registerObject("#" + (i + 1), obj.value, scope);
          }
        }
      env.registerObject("##", objArr, scope);
      env.registerObject("##.typedObject", args, scope);
      }
    else {
      env.unregisterObject("##", scope);
      env.unregisterObject("##.typedObject", scope);
      }
    }
    
  public static void registerAsScopeArguments(GUIKitEnvironment env, 
     String headName, GUIKitTypedObject args[], int scope) throws GUIKitException {
       
    if(args != null) {
      if (headName != null)
        env.registerObject("#0", headName, scope);
      else env.unregisterObject("#0", scope);
      registerAsScopeArguments(env, args, scope);
      }
    else {
      env.unregisterObject("##", scope);
      env.unregisterObject("##.typedObject", scope);
      }
    }
  
  public static GUIKitTypedObject evaluateGUIKitElement(GUIKitEnvironment env, 
      GUIKitTypedObject bean, URL url, Element rootElement, int scope)
  throws GUIKitException {
    GUIKitTypedObject lastResult = null;
    
    if (url != null)
      env.registerObject( MathematicaBSFEngine.ID_SCRIPTCONTEXTURL, url, scope);
    Node node = null;
    // currently Script children
    if (rootElement.getTagName().equals(GUIKitUtils.ELEM_SCRIPT))
      node = rootElement.getFirstChild();  
    else
      node = rootElement;
      
    for(; node != null; node = node.getNextSibling()) {
      GUIKitTypedObject oneResult = env.getDriver().processNode(env, node, bean, url, null);
      lastResult = node.getNodeType() != Node.ELEMENT_NODE ? lastResult : oneResult;
      }

    return lastResult;
    }

  public static GUIKitTypedObject evaluateBSFScript(GUIKitEnvironment env, GUIKitTypedObject bean,
      URL url, Element element, String lang, Object scriptContent) throws GUIKitException  {
    int useScope = MathematicaObjectRegistry.SCOPE_ACTION;
    if(element != null) {
      GUIKitTypedObject[] childResults = env.getDriver().processChildren(env, element, bean, url, null, GUIKitDriver.DEFAULT_RESULTMASK);
      // Under what cases does this get called, should we be adding a scope input
      // to control whether this is always ACTION??
      registerAsScopeArguments(env, childResults, useScope);
      }

    // Should these be always ACTION scope level and only around for this call or useScope?
    if (bean != null)
      env.registerObject(GUIKitUtils.ATTVAL_THIS, bean.value, useScope);
    if (url != null)
      env.registerObject(MathematicaBSFEngine.ID_SCRIPTCONTEXTURL, url, useScope);
    
    Object resultObj = env.evaluateScript(lang, "guikit-document", scriptContent);
    return (GUIKitTypedObject)TypedObjectFactory.create(resultObj);
    }
    
  public static Class[] getCommonArgTypes(Class class1, String s) {
    Method amethod[] = class1.getMethods();
    Class aclass[] = null;

    for(int i = 0; i < amethod.length; i++)
      if(s == null || amethod[i].getName().equalsIgnoreCase(s))
        if(aclass == null) {
          aclass = amethod[i].getParameterTypes();
          }
        else {
          Class aclass1[] = amethod[i].getParameterTypes();
          if(aclass1.length != aclass.length)
              return null;
          for(int j = 0; j < aclass.length; j++)
              if(aclass1[j] != aclass[j])
                  return null;
          }
    return aclass;
    }

  public static String getAttribute(Element element, String attName) {
    String attValue = null;
    Attr attr = element.getAttributeNode(attName);
    if(attr != null)
      attValue = attr.getValue();
    return attValue;
    }

  public static String getChildCharacterData(Element element) {
    if(element == null)
        return null;
    Node node = element.getFirstChild();
    StringBuffer stringbuffer = new StringBuffer();
    for(; node != null; node = node.getNextSibling())
      switch(node.getNodeType()) {
        case Node.ATTRIBUTE_NODE: // '\002'
        default:
            break;

        case Node.TEXT_NODE: // '\003'
        case Node.CDATA_SECTION_NODE: // '\004'
            CharacterData characterdata = (CharacterData)node;
            stringbuffer.append(characterdata.getData());
            break;

        case Node.ELEMENT_NODE: // '\001'
            Element element1 = (Element)node;
            String s = element1.getTagName();
            if(s.equals(ELEM_ARGS))
                stringbuffer = new StringBuffer();
            break;
        }
    return stringbuffer.toString();
    }

  private static String capitalize(String name) {
    if (name.length() > 1) {
      return name.substring(0,1).toUpperCase() + name.substring(1);
      }
    else {
      return name.toUpperCase();
      }
    }
  
  private static class CapitalizedStringsVector extends Vector {
    private static final long serialVersionUID = -1287986975456788648L;
  	public CapitalizedStringsVector() {
  		super();
  		}
  	public CapitalizedStringsVector(int cap) {
  		super(cap);
  		}
		public synchronized boolean add(Object obj) {
			Object useObj = obj;
			if (obj instanceof String) {
        useObj = capitalize((String)obj);
				}
  		super.add(useObj);
  		return true;
  		}
  	}
}
