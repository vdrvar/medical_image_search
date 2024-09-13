#[macro_use] extern crate rocket;
use rocket::fs::{FileServer, relative};
use rocket_dyn_templates::Template;
use std::collections::HashMap;
use std::path::Path;

#[get("/")]
fn index() -> Template {
    let context: HashMap<String, String> = HashMap::new();
    Template::render("index", &context)
}

#[post("/upload", data = "<upload>")]
async fn upload(mut upload: rocket::form::Form<rocket::fs::TempFile<'_>>) -> &'static str {
    // Get the original file name or default to "uploaded_image"
    let file_name = upload.raw_name()
        .and_then(|n| n.as_str())
        .unwrap_or("uploaded_image")
        .to_string();

    // Extract the file extension
    let extension = Path::new(&file_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg"); // Default to jpg if no extension is found

    // Construct the file path with the correct extension
    let file_path = format!("uploaded_image.{}", extension);

    // Persist the uploaded file with the correct extension
    if let Ok(_) = upload.persist_to(&file_path).await {
        "File uploaded successfully!"
    } else {
        "File upload failed."
    }
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", routes![index, upload])
        .mount("/static", FileServer::from(relative!("static")))
        .attach(Template::fairing())  // Attach the template engine
}
