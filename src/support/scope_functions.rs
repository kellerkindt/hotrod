//! These are helper functions which mimic kotlin's scope functions.
//! These helpers shall help write functions as a continuous pipeline:
//! ```rust
//! use hotrod::support::scope_functions::*;
//!
//! let result = vec![1, 2]
//!     .apply(|v| v.push(3))                   // Mutates (&mut self)
//!     .also(|v| println!("Len: {}", v.len())) // Reads (&self)
//!     .r#let(|v| v.into_iter().sum::<i32>()); // Consumes (self), returns i32
//!
//! assert_eq!(result, 6);
//! ```

/// Trait that allows to access a given type immutably before it is returned.
pub trait Also {
    /// Returns `self` but also executes the given `action` on an *immutable* reference of `self`
    /// before.
    fn also(self, action: impl FnOnce(&Self)) -> Self
    where
        Self: Sized;

    /// Returns `self` but also executes the given `action` on an *immutable* reference of `self`
    /// before, if the given `check` value is `true`.
    #[inline]
    fn also_if(self, check: bool, action: impl FnOnce(&Self)) -> Self
    where
        Self: Sized,
    {
        if check {
            self.also(action)
        } else {
            self
        }
    }
}

impl<T> Also for T
where
    T: Sized,
{
    #[inline]
    fn also(self, action: impl FnOnce(&Self)) -> Self
    where
        Self: Sized,
    {
        action(&self);
        self
    }
}

/// Trait that allows to apply changes to a given type mutably before it is returned.
pub trait Apply {
    /// Returns `self` but applies the given `action` on a *mutable* reference of `self` before.
    fn apply(self, action: impl FnOnce(&mut Self)) -> Self
    where
        Self: Sized;

    /// Returns `self` but applies the given `action` on a *mutable* reference of `self` before, if
    /// the given `check` value is `true`.
    #[inline]
    fn apply_if(self, check: bool, action: impl FnOnce(&mut Self)) -> Self
    where
        Self: Sized,
    {
        if check {
            self.apply(action)
        } else {
            self
        }
    }
}

impl<T> Apply for T
where
    T: Sized,
{
    #[inline]
    fn apply(mut self, action: impl FnOnce(&mut Self)) -> Self
    where
        Self: Sized,
    {
        action(&mut self);
        self
    }
}

/// Trait that allows to map one type to another.
pub trait Let {
    /// Lets you map `self` to a new type via the given `mapping` function and returns the new value.
    fn r#let<R>(self, mapping: impl FnOnce(Self) -> R) -> R
    where
        Self: Sized,
        R: Sized;
}

impl<T> Let for T
where
    T: Sized,
{
    #[inline]
    fn r#let<R>(self, mapping: impl FnOnce(Self) -> R) -> R
    where
        Self: Sized,
        R: Sized,
    {
        mapping(self)
    }
}
