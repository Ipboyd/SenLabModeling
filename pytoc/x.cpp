#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic\include\types\float64.hpp>
#include <pythonic\types\float64.hpp>
#include <pythonic/include/operator_/add.hpp>
#include <pythonic/operator_/add.hpp>
namespace 
{
  namespace __pythran_x
  {
    struct add2
    {
      typedef void callable;
      typedef void pure;
      template <typename argument_type0 , typename argument_type1 >
      struct type
      {
        typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
        typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type1;
        typedef decltype(pythonic::operator_::add(std::declval<__type0>(), std::declval<__type1>())) __type2;
        typedef typename pythonic::returnable<__type2>::type __type3;
        typedef __type3 result_type;
      }  
      ;
      template <typename argument_type0 , typename argument_type1 >
      inline
      typename type<argument_type0, argument_type1>::result_type operator()(argument_type0 p, argument_type1 q) const
      ;
    }  ;
    template <typename argument_type0 , typename argument_type1 >
    inline
    typename add2::type<argument_type0, argument_type1>::result_type add2::operator()(argument_type0 p, argument_type1 q) const
    {
      return pythonic::operator_::add(p, q);
    }
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
inline
typename __pythran_x::add2::type<double, double>::result_type add20(double&& p, double&& q) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_x::add2()(p, q);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap_add20(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[2+1];
    
    char const* keywords[] = {"p", "q",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OO",
                                     (char**)keywords , &args_obj[0], &args_obj[1]))
        return nullptr;
    if(is_convertible<double>(args_obj[0]) && is_convertible<double>(args_obj[1]))
        return to_python(add20(from_python<double>(args_obj[0]), from_python<double>(args_obj[1])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall_add2(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_add20(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "add2", "\n""    - add2(float64, float64)", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "add2",
    (PyCFunction)__pythran_wrapall_add2,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - add2(float64, float64)"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "x",            /* m_name */
  "",         /* m_doc */
  -1,                  /* m_size */
  Methods,             /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};
PyMODINIT_FUNC
PyInit_x(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
#if defined(GNUC) && !defined(__clang__)
__attribute__ ((externally_visible))
#endif
#endif
;
PyMODINIT_FUNC
PyInit_x(void) {
    import_array();

    PyObject* theModule = PyModule_Create(&moduledef);
    if(! theModule)
        return theModule;

                #ifdef Py_GIL_DISABLED
                    PyUnstable_Module_SetGIL(theModule, Py_MOD_GIL_NOT_USED);
                #endif
    PyObject * theDoc = Py_BuildValue("(ss)",
                                      "0.18.0",
                                      "9ce0fa0e6abbf94522aa26a39988a39de75fbb5f8738ad111d05fb7ad29d80c2");
    if(! theDoc)
        return theModule;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    return theModule;
}

#endif